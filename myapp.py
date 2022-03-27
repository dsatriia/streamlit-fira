import streamlit as st
import os
os.environ["LI_AT_COOKIE"] = "AQEDATnUOO0BfYygAAABf776ZHoAAAF_4wboelYAKm0H0j6x3xT0juMh9Xb_K8jwjwrXeZoUcO_m67Rde0hI9o85m56yc55lZJeatodO76iIdGjGOOjx4CEdL68meJDJ02UKObv6xSgX8fxjqiQHDs_g"

#packages input cv 
import csv #Dengan modul csv untuk beralih ke suatu baris dan mengaksesnya

#packages translate
from google_trans_new import google_translator

#packages database
import sqlite3
import hashlib

#packages browser
from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.chrome.options import Options

#packages untuk menampilkan gambar dan logo
from PIL import Image

#packages Load LSA
import gensim
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import LsiModel
from gensim import corpora, similarities
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re
import pandas as pd
import nltk

# Packages Scraping
from linkedin_jobs_scraper.filters import (
	RelevanceFilters,
	TimeFilters,
	TypeFilters,
	ExperienceLevelFilters,
)
from linkedin_jobs_scraper.query import Query, QueryOptions, QueryFilters
from linkedin_jobs_scraper.events import Events, EventData
from linkedin_jobs_scraper import LinkedinScraper
import logging

translator = google_translator()

# Packages Pra-Proses
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")
stop_words = set(stopwords.words("english"))
# print(stop_words)

#session login
if 'isSearched' not in st.session_state:
    st.session_state['isSearched'] = False

#session fitur upload cv dll
if 'isUser' not in st.session_state:
    st.session_state['isUser'] = False

#session login admin
if 'isAdmin' not in st.session_state:
    st.session_state['isAdmin'] = False

#session download cv setelah diedit
if 'isCvDataUpdated' not in st.session_state:
    st.session_state['isCvDataUpdated'] = False

# variable contains all countries name in the world
countries_name = ["Afghanistan", "Albania", "Algeria", "American Samoa", "Andorra", "Angola", "Anguilla", "Antarctica", "Antigua and Barbuda", "Argentina", "Armenia", "Aruba", "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bermuda", "Bhutan", "Bolivia", "Bosnia and Herzegowina", "Botswana", "Bouvet Island", "Brazil", "British Indian Ocean Territory", "Brunei Darussalam", "Bulgaria", "Burkina Faso", "Burundi", "Cambodia", "Cameroon", "Canada", "Cape Verde", "Cayman Islands", "Central African Republic", "Chad", "Chile", "China", "Christmas Island", "Cocos (Keeling) Islands", "Colombia", "Comoros", "Congo", "Congo, the Democratic Republic of the", "Cook Islands", "Costa Rica", "Cote d'Ivoire", "Croatia (Hrvatska)", "Cuba", "Cyprus", "Czech Republic", "Denmark", "Djibouti", "Dominica", "Dominican Republic", "East Timor", "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", "Ethiopia", "Falkland Islands (Malvinas)", "Faroe Islands", "Fiji", "Finland", "France", "France Metropolitan", "French Guiana", "French Polynesia", "French Southern Territories", "Gabon", "Gambia", "Georgia", "Germany", "Ghana", "Gibraltar", "Greece", "Greenland", "Grenada", "Guadeloupe", "Guam", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana", "Haiti", "Heard and Mc Donald Islands", "Holy See (Vatican City State)", "Honduras", "Hong Kong", "Hungary", "Iceland", "India", "Indonesia", "Iran (Islamic Republic of)", "Iraq", "Ireland", "Israel", "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati", "Korea, Democratic People's Republic of", "Korea, Republic of", "Kuwait", "Kyrgyzstan", "Lao, People's Democratic Republic", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libyan Arab Jamahiriya", "Liechtenstein", "Lithuania", "Luxembourg", "Macau", "Macedonia, The Former Yugoslav Republic of", "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Marshall Islands", "Martinique", "Mauritania", "Mauritius", "Mayotte", "Mexico", "Micronesia, Federated States of", "Moldova, Republic of", "Monaco", "Mongolia", "Montserrat", "Morocco", "Mozambique", "Myanmar", "Namibia", "Nauru", "Nepal", "Netherlands", "Netherlands Antilles", "New Caledonia", "New Zealand", "Nicaragua", "Niger", "Nigeria", "Niue", "Norfolk Island", "Northern Mariana Islands", "Norway", "Oman", "Pakistan", "Palau", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Pitcairn", "Poland", "Portugal", "Puerto Rico", "Qatar", "Reunion", "Romania", "Russian Federation", "Rwanda", "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa", "San Marino", "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Seychelles", "Sierra Leone", "Singapore", "Slovakia (Slovak Republic)", "Slovenia", "Solomon Islands", "Somalia", "South Africa", "South Georgia and the South Sandwich Islands", "Spain", "Sri Lanka", "St. Helena", "St. Pierre and Miquelon", "Sudan", "Suriname", "Svalbard and Jan Mayen Islands", "Swaziland", "Sweden", "Switzerland", "Syrian Arab Republic", "Taiwan, Province of China", "Tajikistan", "Tanzania, United Republic of", "Thailand", "Togo", "Tokelau", "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", "Turks and Caicos Islands", "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates", "United Kingdom", "United States", "United States Minor Outlying Islands", "Uruguay", "Uzbekistan", "Vanuatu", "Venezuela", "Vietnam", "Virgin Islands (British)", "Virgin Islands (U.S.)", "Wallis and Futuna Islands", "Western Sahara", "Yemen", "Yugoslavia", "Zambia", "Zimbabwe"]

options = Options()
def get_chromedriver_path():
	results = glob.glob(
		"/**/chromedriver", recursive=True
	)  # workaround on streamlit sharing
	which = results[0]
	return which

# Load Our Dataset
def load_data(data):
	df = pd.read_csv(data)
	return df

#topic coherence
from gensim.models import CoherenceModel

def local_css(file_name):
	with open(file_name) as f:
		st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# passlib,hashlib,bcrypt,scrypt
def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False

# DB Management
conn = sqlite3.connect("data.db")
c = conn.cursor()

# DB  Functions
def create_usertable():
	c.execute("CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)")

def is_user_exist(username):
	c.execute("SELECT * FROM userstable WHERE username = ?", (username,))
	data = c.fetchall()
	return data

def delete_userdata(username):

	if len(is_user_exist(username)) < 1:
		return False
	c.execute("DELETE FROM userstable WHERE username = ?", (username,))
	conn.commit()
	return True

def add_userdata(username, password):

	if is_user_exist(username):
		return False
	c.execute(
		"INSERT INTO userstable(username,password,role) VALUES (?,?,?)", (
			username, password, "user")
	)
	conn.commit()
	return True


def add_admindata(username, password):
	if is_user_exist(username):
		return False
	c.execute(
		"INSERT INTO userstable(username,password,role) VALUES (?,?,?)", (
			username, password, "admin")
	)
	conn.commit()
	return True


def login_user(username, password):
	c.execute(
		"SELECT * FROM userstable WHERE username =? AND password = ? AND role = 'user'",
		(username, password),
	)
	data = c.fetchall()	
	return data


def login_admin(username, password):
	c.execute(
		"SELECT * FROM userstable WHERE username =? AND password = ? AND role = 'admin'",
		(username, password),
	)
	data = c.fetchall()
	return data

def view_all_users():
	c.execute("SELECT * FROM userstable")
	data = c.fetchall()
	return data

def main():
	"""Login"""
	# st.title("Welcome")

	menu = ["Home", "SignUp", "About", "Administrator"]
	choice = st.selectbox("Menu", menu)

	isSearched = st.session_state.isSearched
	isCvDataUpdated = st.session_state.isCvDataUpdated

	if choice == "Home":
		st.title("Login Here")
			
		username = st.text_input("User Name")
		password = st.text_input("Password", type="password")
		result = False

		if st.button("Login"):
			create_usertable()
			hashed_pswd = make_hashes(password)
			result = login_user(username, check_hashes(password, hashed_pswd))
			
		if result:
			st.session_state.isUser = True

		if st.session_state.isUser == True:
			st.success("Logged In as {}".format(username))		
			
			st.subheader("Recommendation System Guide")
			video_file = open('tutorial.mp4', 'rb')
			video_bytes = video_file.read()
			st.video(video_bytes)
			
			task = st.selectbox(
# 				"Task", ["Recommend", "TemplateCV", "Download Data Hasil Scrap"])
				"Menu", ["Recommend", "CV Template"])

			if task == "Recommend":
				st.title("Job Recommender")
				# countries = []	
				st.markdown("""---""")
				st.title("Update Linkedin Ads")
				st.subheader("**Allows to input multiple countries**")
				countries = st.multiselect('Countries name',countries_name,[])
				job_title = st.text_input("Input Job Title")
				jum = st.number_input("Input number of Ads", 2, 100, 3) 		
				
				iterate_number = []
				if st.button("Perbarui (Test)"):
					id = []
					post_title = []
					company_name = []
					post_date = []
					job_location = []
					job_des = []
					link = []
# 					total_employees = []
# 					actively_recruiting = []
						
					# countries = countries_string.split(",")
					total_countries = len(countries)	
					if total_countries == 0:
						st.error("Please select at least one country.")
						return
					
					# countries_string = countries_string.replace(" ", "")	
					if total_countries > jum :  #minimal 1 negara 1 iklan yg dicari. misal 4 negara input dengan pencarian 3 maka muncul warning
						st.warning("The number of countries entered is more than the number of desired ads.")
					
					for i in range(total_countries):
						temp = int(jum/total_countries)
						iterate_number.append(temp)

					sisa = jum%total_countries

					for i in range(sisa):
						iterate_number[i] += 1

					st.write(countries)	
					st.write(iterate_number)
					# st.write(sisa)
								
					# foreach iterate_number
					id = []
					post_title = []
					company_name = []
					post_date = []
					job_location = []
					job_des = []
					link = []
					total_employees = []
					actively_recruiting = []

					for i in range(len(iterate_number)):
											
						try:
							# Change root logger level (default is WARN)
							logging.basicConfig(level=logging.INFO)							

							def on_data(data: EventData):
								print(
									"[ON_DATA]",
									data.title,
									data.company,
									data.date,
									data.description,
									data.link,
									len(data.description),
								)
								post_title.append(
									translator.translate(
										data.title, lang_src="auto", lang_tgt="en"
									)
								)
								# post_title.append(data.title)
								id_job = len(post_title)
								id.append(id_job)
								job_location.append(data.place)
								company_name.append(
									translator.translate(
										data.company, lang_src="auto", lang_tgt="en"
									)
								)
								# company_name.append(data.company)
								post_date.append(data.date)
								job_desc = translator.translate(
									data.description, lang_src="auto", lang_tgt="en"
								)
								job_des.append(job_desc)
								link.append(data.link)
# 								total_employees.append(data.total_employees)
# 								actively_recruiting.append(data.actively_recruiting)
								# print(data.description)
								# print(job_desc)

							def on_error(error):
								print("[ON_ERROR]", error)

							def on_end():
								print("[ON_END]")

							scraper = LinkedinScraper(
								# Custom Chrome executable path (e.g. /foo/bar/bin/chromedriver)
								chrome_executable_path="chromedriver",
								chrome_options=None,  # Custom Chrome options here
								headless=True,  # Overrides headless mode only if chrome_options is None
								# How many threads will be spawned to run queries concurrently (one Chrome driver for each thread)
								max_workers=1,
								# Slow down the scraper to avoid 'Too many requests (429)' errors
								slow_mo=1,
							)

							# Add event listeners
							scraper.on(Events.DATA, on_data)
							scraper.on(Events.ERROR, on_error)
							scraper.on(Events.END, on_end)

							queries = [
								Query(
									query=job_title,
									options=QueryOptions(
										locations=countries[i],
										optimize=False,
										limit=int(iterate_number[i]),
										filters=QueryFilters(
											relevance=RelevanceFilters.RECENT,
										),
									),
								)
							]

							scraper.run(queries)
							
						except:
							results = "Not Found"
					
					job_data = pd.DataFrame(
						{
							"Job_ID": id,
							"Date": post_date,
							"Company Name": company_name,									
# 							"Total Employees": total_employees,
# 							"Actively Recruiting": actively_recruiting,
							"Job_Title": post_title,
							"Location": job_location,
							"Description": job_des,
							"Link": link,
						}
					)

					# cleaning description column
					job_data["Description"] = job_data[
						"Description"
					].str.replace("\n", " ")

					# print(job_data.info())
					st.subheader("Data Hasil Scrap")
					# job_data.head()
					job_data.to_csv(
						"datascraptest.csv", index=0, encoding="utf-8"
					)
					dframe = load_data("datascraptest.csv")
					st.dataframe(dframe.head(10))
				
					st.session_state.isSearched = True
					isSearched = st.session_state.isSearched

				st.subheader("Filter Job")

				if isSearched == True:
					filter_jobtype = [
						None,
						TypeFilters.FULL_TIME,
						TypeFilters.PART_TIME,
						TypeFilters.TEMPORARY,
						TypeFilters.CONTRACT,
					]
					jobtype = st.selectbox("Job_Type", filter_jobtype)
					
					filter_time = [
						None,
						TimeFilters.DAY,
						TimeFilters.WEEK,
						TimeFilters.MONTH,
						TimeFilters.ANY,
					]
					time_iklan = st.selectbox("Date Posted", filter_time)

# 					filter_actively_recruiting = [
# 						None,
# 						"Yes",
# 						"No"
# 					]
# 					selected_filter_actively_recruiting = st.selectbox("Actively Recruiting", filter_actively_recruiting)

					if st.button("Perbarui Iklan"):  #jika tidak ada iklan yg dicari maka akan menampilkan iklan yang direkomendasikan/mungkin kita interest
						id = []
						post_title = []
						company_name = []
						post_date = []
						job_location = []
						job_des = []
						link = []
# 						total_employees = []
# 						actively_recruiting = []
						
						# Fungsi len() digunakan untuk mengetahui panjang (jumlah item atau anggota) dari objek
						total_countries = len(countries)				
						if total_countries == 0:
							st.error("Please select at least one country.")
							return
						if total_countries > jum :  #minimal 1 negara 1 iklan yg dicari. misal 4 negara input dengan pencarian 3 maka muncul warning
							st.warning("The number of countries entered is more than the number of desired ads.")
						
						for i in range(total_countries):
							temp = int(jum/total_countries)
							iterate_number.append(temp)

						sisa = jum%total_countries

						for i in range(sisa):
							iterate_number[i] += 1
											
						for i in range(len(iterate_number)):
							
							try:
								# Change root logger level (default is WARN)
								logging.basicConfig(level=logging.INFO)	

								def on_data(data: EventData):
									
# 									if selected_filter_actively_recruiting == None:
										post_title.append(
											translator.translate(
												data.title, lang_src="auto", lang_tgt="en"
											)
										)
										# 								post_title.append(data.title)
										id_job = len(post_title)
										id.append(id_job)
										job_location.append(data.place)
										company_name.append(
											translator.translate(
												data.company, lang_src="auto", lang_tgt="en"
											)
										)
										# 								company_name.append(data.company)
										post_date.append(data.date)
										job_des.append(
											translator.translate(
												data.description, lang_src="auto", lang_tgt="en"
											)
										)
										# 								job_des.append(data.description)
										link.append(data.link)
# 										total_employees.append(data.total_employees)
# 										actively_recruiting.append(data.actively_recruiting)
									
# 									elif data.actively_recruiting == selected_filter_actively_recruiting:
# 										post_title.append(
# 											translator.translate(
# 												data.title, lang_src="auto", lang_tgt="en"
# 											)
# 										)
# 										# 								post_title.append(data.title)
# 										id_job = len(post_title)
# 										id.append(id_job)
# 										job_location.append(data.place)
# 										company_name.append(
# 											translator.translate(
# 												data.company, lang_src="auto", lang_tgt="en"
# 											)
# 										)
# 										# 								company_name.append(data.company)
# 										post_date.append(data.date)
# 										job_des.append(
# 											translator.translate(
# 												data.description, lang_src="auto", lang_tgt="en"
# 											)
# 										)
# 										# 								job_des.append(data.description)
# 										link.append(data.link)
# # 										total_employees.append(data.total_employees)
# # 										actively_recruiting.append(data.actively_recruiting)

								def on_error(error):
									print("[ON_ERROR]", error)

								def on_end():
									print("[ON_END]")

								scraper = LinkedinScraper(
									# Custom Chrome executable path (e.g. /foo/bar/bin/chromedriver)
									chrome_executable_path="chromedriver",
									chrome_options=None,  # Custom Chrome options here
									headless=True,  # Overrides headless mode only if chrome_options is None
									# How many threads will be spawned to run queries concurrently (one Chrome driver for each thread)
									max_workers=1,
									# Slow down the scraper to avoid 'Too many requests (429)' errors
									slow_mo=1,
								)

								# Add event listeners
								scraper.on(Events.DATA, on_data)
								scraper.on(Events.ERROR, on_error)
								scraper.on(Events.END, on_end)

								queries = [
									Query(
										query=job_title,
										options=QueryOptions(
											locations=countries[i],
											optimize=False,
											limit=int(iterate_number[i]),
											filters=QueryFilters(										
												relevance=RelevanceFilters.RECENT,
												type=jobtype,
												time=time_iklan,
											),
										),
									)
								]

								scraper.run(queries)

							except:
								results = "Not Found"
							
						job_data = pd.DataFrame(
							{
								"Job_ID": id,
								"Date": post_date,
								"Company Name": company_name,							
# 								"Total Employees": total_employees,
# 								"Actively Recruiting": actively_recruiting,
								"Job_Title": post_title,
								"Location": job_location,
								"Description": job_des,
								"Link": link,
							}
						)

						# cleaning description column
						job_data["Description"] = job_data[
							"Description"
						].str.replace("\n", " ")

						# print(job_data.info())
						st.subheader("Data Hasil Scrap")
						# job_data.head()
						job_data.to_csv(
							"datascraptest.csv", index=0, encoding="utf-8"
						)
						dframe = load_data("datascraptest.csv")
						st.dataframe(dframe.head(10))
				else:
					st.error("Please update the ad first.")
					
				if isSearched == True:
					# st.subheader("CV Writing Guide")
					# image = Image.open("panduanedit.png")
					# st.image(image, caption="Format CV")
					st.subheader("Update CV Data")		
					education_value =""		
					pic1 = Image.open("templateexample.jpg")
					st.image(pic1, caption="example")
					education_section = st.text_area('Education Section:',education_value,200)
					
					experience_value =" "		
					experience_section = st.text_area('Experience Section:',experience_value,200)
					

					skill_value =" "
					skill_section = st.text_area('Skill Section:',skill_value,200)
					
					if st.button("Update Data"):
						cv_desc_value = "Education Section: " + education_section + ", " + "Experience Section: " + experience_section + ", " + "Skills Section: " + skill_section
						st.warning(cv_desc_value)

						st.success("CV Data Updated!")
						st.session_state.isCvDataUpdated = True
						isCvDataUpdated = st.session_state.isCvDataUpdated	

						cv_desc_value = "Education Section: " + education_section + ", " + "Experience Section: " + experience_section + ", " + "Skills Section: " + skill_section
						# csv header
						fieldnames = ['cv_desc']

						# csv data
						rows = [
							{'cv_desc': cv_desc_value}
						]

						with open('templatecv.csv', 'w', encoding='UTF8', newline='') as f:
							writer = csv.DictWriter(f, fieldnames=fieldnames)
							writer.writeheader()
							writer.writerows(rows)

# 					st.subheader("Download CV ")		
# 					if isCvDataUpdated == True:		
# 						st.write("Click the button to download CV template")	

# 						with open("templatecv.csv", "rb") as file:  #Read the file in Binary mode : untuk buka doc, biner : bahasa mesin. wb: untuk menulis

# 							st.download_button(
# 								label="Download",
# 								data=file,
# 								file_name="template.csv",
# 								mime="text/csv",
# 							)				

# 					else:
# 						st.error("Please update CV data first.")
			
# 				st.subheader(
# 					"Upload CV to Find Job Ad Recommendations")
# 				# st.write(st.session_state.isSearched)
# 				if isSearched == True:
					jumlah = st.number_input(
						"Input Banyak Iklan yang ingin Ditampilkan", 2, 100, 3
					)  # mulai,max,default
					if st.button("Temukan Iklan yang Cocok"):
						try:

							def remove_punc(text):
								symbols = (
									r"â€¢!\"#$%&()*+-.,/:;<=>?@[\]^_`'{|}~\\0123456789"
								)
								output = [
									char for char in text if char not in symbols]
								return "".join(output)

							def stemSentence(tokens):
								# token_words=word_tokenize(text)

								porter = PorterStemmer()
								stem_sentence = []
								for word in tokens:
									stem_sentence.append(porter.stem(word))
								return stem_sentence

							def lemmatization(tokens):
								lemmatizer = WordNetLemmatizer()
								return [lemmatizer.lemmatize(word) for word in tokens]

							def stopwordSentence(tokens):
								return [
									word for word in tokens if word not in stop_words
								]

							def caseFold(text):
								return text.lower()

							def preProcessPipeline(text, print_output=False):
								if print_output:
									print("Teks awal:")
									print(text)
								text = remove_punc(text)
								if print_output:
									print("Setelah menghilangkan tanda baca:")
									print(text)

								text = caseFold(text)
								if print_output:
									print("Setelah Casefold")
									print(text)

								token_words = word_tokenize(text)
								token_words = lemmatization(token_words)
								if print_output:
									print("Setelah lemmatization:")
									print(" ".join(token_words))

								token_words = stopwordSentence(token_words)
								if print_output:
									print("Setelah menghilangkan stopwords:")
									print(" ".join(token_words))

								return " ".join(token_words)

							documents_train = pd.read_csv(
								"datascraptest.csv", error_bad_lines=False
							)
							train_text = documents_train["Description"].apply(
								preProcessPipeline
							)
							documents_test = pd.read_csv(
								"templatecv.csv", error_bad_lines=False)
							test_text = documents_test["cv_desc"].apply(
								preProcessPipeline
							)

							nltk_tokens = [nltk.word_tokenize(
								i) for i in train_text]
							y = nltk_tokens

							dictionary = gensim.corpora.Dictionary(y)
							text = y

							corpus = [dictionary.doc2bow(i) for i in text]
							tfidf = gensim.models.TfidfModel(
								corpus, smartirs="npu")
							corpus_tfidf = tfidf[corpus]

							#bmemilih num_topics optimal sesuai topic coherence
							num_topics = list(range(1,10))
							coherence_scores = []

							for i in num_topics:
								lsa_model = LsiModel(corpus=corpus, num_topics=i, id2word = dictionary)
								coherence_model = CoherenceModel(model=lsa_model, texts=y, dictionary=dictionary, coherence='c_v')
								coherence_lsa = coherence_model.get_coherence()

								coherence_scores.append(coherence_lsa)
							
# 							for m, cv in zip(num_topics, coherence_scores):
#     								st.write("Num Topics =", m, "has Coherence Value of", round(cv, 3))
							#short paling besar input ke num_topics=bigest

							#mengambil nilai dalam array
							max_Coherence = np.argmax(coherence_scores)
							
# 							st.write("numb of topic:",max_Coherence)
# 							st.write("best coherence score:",coherence_scores[max_Coherence])

							lsi_model = LsiModel(
								corpus=corpus_tfidf, id2word=dictionary, num_topics=max_Coherence
							)
							print(
								"Derivation of Term Matrix T of Training Document Word Stems: ",
								lsi_model.get_topics(),
							)
							# Derivation of Term Document Matrix of Training Document Word Stems = M' x [Derivation of T]
							print(
								"LSI Vectors of Training Document Word Stems: ",
								[
									lsi_model[document_word_stems]
									for document_word_stems in corpus
								],
							)
							cosine_similarity_matrix = similarities.MatrixSimilarity(
								lsi_model[corpus]
							)

							word_tokenizer = nltk.tokenize.WordPunctTokenizer()
							words = word_tokenizer.tokenize(test_text[0])

							vector_lsi_test = lsi_model[dictionary.doc2bow(
								words)]

							cosine_similarities_test = cosine_similarity_matrix[
								vector_lsi_test
							]
						
							most_similar_document_test = train_text[
								np.argmax(cosine_similarities_test)
							]
							 							
							#Mengubah nilai cosine float ke persentase
							cst1 = cosine_similarities_test*int(100)
							cst = cst1.astype(int)

							cst_terurut = sorted(
								cosine_similarities_test, reverse=True)

							iklan = cosine_similarities_test.argsort(
							)[-jumlah:][::-1]

							def generator_cosines(iklan):
								for i in iklan:
									yield i

							# print data awal di csv
							for i in iklan:
								st.write(
									"Similarity Level Between CV and Ads :", f"{cst[i]}","%"
								)
								st.write(
									"Post Date :", f"{documents_train['Date'][i]}\n"
								)
								st.write(
									"Company Name :",
									f"{documents_train['Company Name'][i]}\n",
								)
								st.write(
									"Job Title :",
									f"{documents_train['Job_Title'][i]}\n",
								)
								st.write(
									"Job Description :",
									f"{documents_train['Description'][i]}\n",
								)
								st.write(
									"Job Location :", f"{documents_train['Location'][i]}\n"
								)
								st.write(
									"Link Ads :",
									f"{documents_train['Link'][i]}\n",
								)
					
								st.markdown("""---""")

						except:
							results = "Not Found"
				else:
					st.error("Please update the ad first.")

			elif task == "CV Template":
				st.subheader("Download CV ")
					# st.subheader("CV Writing Guide")
					# image = Image.open("panduanedit.png")
					# st.image(image, caption="Format CV")
				st.subheader("Update CV Data")		
				education_value =""		
				pic1 = Image.open("templateexample.jpg")
				st.image(pic1, caption="example")
				education_section = st.text_area('Education Section:',education_value,200)
			
				experience_value =" "		
				experience_section = st.text_area('Experience Section:',experience_value,500)
				
				skill_value =" "
				skill_section = st.text_area('Skill Section:',skill_value,200)
				
				if st.button("Update Data"):
					cv_desc_value = "Education Section: " + education_section + ", " + "Experience Section: " + experience_section + ", " + "Skills Section: " + skill_section
					st.warning(cv_desc_value)

					st.success("CV Data Updated!")
					st.session_state.isCvDataUpdated = True
					isCvDataUpdated = st.session_state.isCvDataUpdated	

					cv_desc_value = "Education Section: " + education_section + ", " + "Experience Section: " + experience_section + ", " + "Skills Section: " + skill_section
					# csv header
					fieldnames = ['cv_desc']

					# csv data
					rows = [
						{'cv_desc': cv_desc_value}
					]

					with open('templatecv.csv', 'w', encoding='UTF8', newline='') as f:
						writer = csv.DictWriter(f, fieldnames=fieldnames)
						writer.writeheader()
						writer.writerows(rows)

				st.subheader("Download Template CV ")		
				if isCvDataUpdated == True:		
					st.write("Click the button to download CV template")	

					with open("templatecv.csv", "rb") as file:

						st.download_button(
							label="Download",
							data=file,
							file_name="template.csv",
							mime="text/csv",
						)				
					st.subheader("CV Writing Guide")
					image = Image.open("panduanedit.png")
					st.image(image, caption="Format CV")
				else:
					st.error("Please update CV data first.")
				
		else:
			st.warning("Incorrect Username/Password")
		
		image = Image.open("logoku.png")
		st.image(image, width=500)			

	elif choice == "SignUp":
		st.title("Create New Account")
		new_user = st.text_input("Username")
		new_password = st.text_input("Password", type="password")

		if st.button("Signup"):
			create_usertable()
			if add_userdata(new_user, make_hashes(new_password)):
				st.success("You have successfully created a valid Account")
				st.info("Go to Login Menu to login")
			else:
				st.warning("Username already exist")

	elif choice == "About":
		st.title("About")
		st.subheader("Job Recommender System")
		st.write("Get job recommendations that match your curriculum vitae")

	elif choice == "Administrator":
		username = st.text_input("Username Admin")
		password = st.text_input("Password Admin", type="password")
		result_admin = False    #inisialisasi awal / value awal
		result_user = False

		st.title("Profiles")

		if st.button("Login Admin"):
			create_usertable()
			hashed_pswd = make_hashes(password)
			result_admin = login_admin(
				username, check_hashes(password, hashed_pswd))
			result_user = login_user(
				username, check_hashes(password, hashed_pswd))
		
		if result_user:
			st.session_state.isUser = True

		if result_admin:
			st.session_state.isAdmin = True

		if st.session_state.isAdmin == True:
			task = st.selectbox(
				"Menu", ["Profiles", "Download Ad Search Results"])

			if task == "Profiles":
				st.subheader("Add Admin")
				new_admin_username = st.text_input("New Username")
				new_admin_password = st.text_input(
					"New Password", type="password")

				if st.button("Add new Admin"):
					if add_admindata(new_admin_username, make_hashes(new_admin_password)):
						st.success(
							"You have successfully created a valid Admin Account")
					else:
						st.warning("Username already exist")
				st.markdown("""---""")

				st.subheader("Delete User")
				username = st.text_input("Username")

				if st.button("Delete User"):
					if delete_userdata(username):
						st.success(
							"You have successfully delete user : "+username)
					else:
						st.warning("Username not found")
				st.markdown("""---""")

				st.subheader("User Profiles")
				user_result = view_all_users()
				clean_db = pd.DataFrame(
					user_result, columns=["Username", "Password", "Role"]
				)
				st.dataframe(clean_db)

			elif task == "Download Ad Search Results":
				
					st.title("Job Recommender")
					st.markdown("""---""")
					st.title("Update Linkedin Ads")
					st.subheader("**Allows to Input Multiple Countries**")
					countries = st.multiselect('Countries name',countries_name,[])
					job_title = st.text_input("Input Job Title")
					jum = st.number_input("Input number of Ads", 2, 100, 5) 	#jum adalah banyak iklan yg ingin dimunculkan	

					iterate_number = []
					if st.button("Perbarui (Test)"):
						id = []
						post_title = []
						company_name = []
						post_date = []
						job_location = []
						job_des = []
						link = []
	# 					total_employees = []
	# 					actively_recruiting = []

						total_countries = len(countries) #len : mengembalikan panjang sebuah string				
						if total_countries > jum :  #minimal 1 negara 1 iklan yg dicari. misal 4 negara input dengan pencarian 3 maka muncul warning
							st.warning("The number of countries entered is more than the number of desired ads.")

						for i in range(total_countries):
							temp = int(jum/total_countries)
							iterate_number.append(temp)

						sisa = jum%total_countries

						for i in range(sisa):
							iterate_number[i] += 1

						st.write(countries)	
						st.write(iterate_number)
						
						# foreach iterate_number
						id = []
						post_title = []
						company_name = []
						post_date = []
						job_location = []
						job_des = []
						link = []
						total_employees = []
						actively_recruiting = []

						for i in range(len(iterate_number)):

							try:
								# Change root logger level (default is WARN)
								logging.basicConfig(level=logging.INFO)							

								def on_data(data: EventData):
									print(
										"[ON_DATA]",
										data.title,
										data.company,
										data.date,
										data.description,
										data.link,
										len(data.description),
									)
									post_title.append(
										translator.translate(
											data.title, lang_src="auto", lang_tgt="en"
										)
									)
									# 								post_title.append(data.title)
									id_job = len(post_title)
									id.append(id_job)
									job_location.append(data.place)
									company_name.append(
										translator.translate(
											data.company, lang_src="auto", lang_tgt="en"
										)
									)
									# 								company_name.append(data.company)
									post_date.append(data.date)
									job_desc = translator.translate(
										data.description, lang_src="auto", lang_tgt="en"
									)
									job_des.append(job_desc)
									link.append(data.link)
	# 								total_employees.append(data.total_employees)
	# 								actively_recruiting.append(data.actively_recruiting)
									# print(data.description)
									# print(job_desc)

								def on_error(error):
									print("[ON_ERROR]", error)

								def on_end():
									print("[ON_END]")

								scraper = LinkedinScraper(
									# Custom Chrome executable path (e.g. /foo/bar/bin/chromedriver)
									chrome_executable_path="chromedriver",
									chrome_options=None,  # Custom Chrome options here
									headless=True,  # Overrides headless mode only if chrome_options is None
									# How many threads will be spawned to run queries concurrently (one Chrome driver for each thread)
									max_workers=1,
									# Slow down the scraper to avoid 'Too many requests (429)' errors
									slow_mo=1,
								)

								# Add event listeners
								scraper.on(Events.DATA, on_data)
								scraper.on(Events.ERROR, on_error)
								scraper.on(Events.END, on_end)

								queries = [
									Query(
										query=job_title,
										options=QueryOptions(
											locations=countries[i],
											optimize=False,
											limit=int(iterate_number[i]),
											filters=QueryFilters(
												relevance=RelevanceFilters.RECENT,
											),
										),
									)
								]

								scraper.run(queries)

							except:
								results = "Not Found"

						job_data = pd.DataFrame(
							{
								"Job_ID": id,
								"Date": post_date,
								"Company Name": company_name,									
	# 							"Total Employees": total_employees,
	# 							"Actively Recruiting": actively_recruiting,
								"Job_Title": post_title,
								"Location": job_location,
								"Description": job_des,
								"Link": link,
							}
						)

						# cleaning description column
						job_data["Description"] = job_data[
							"Description"
						].str.replace("\n", " ")

						# print(job_data.info())
						st.subheader("Data Hasil Scrap")
						# job_data.head()
						job_data.to_csv(
							"datascraptest.csv", index=0, encoding="utf-8"
						)
						dframe = load_data("datascraptest.csv")
						st.dataframe(dframe.head(10))

						st.session_state.isSearched = True
						isSearched = st.session_state.isSearched
						
					if isSearched == True:
						filter_jobtype = [
							None,
							TypeFilters.FULL_TIME,
							TypeFilters.PART_TIME,
							TypeFilters.TEMPORARY,
							TypeFilters.CONTRACT,
						]
						jobtype = st.selectbox("Job_Type", filter_jobtype)

						filter_time = [
							None,
							TimeFilters.DAY,
							TimeFilters.WEEK,
							TimeFilters.MONTH,
							TimeFilters.ANY,
						]
						time_iklan = st.selectbox("Date Posted", filter_time)

	# 					filter_actively_recruiting = [
	# 						None,
	# 						"Yes",
	# 						"No"
	# 					]
	# 					selected_filter_actively_recruiting = st.selectbox("Actively Recruiting", filter_actively_recruiting)

						if st.button("Perbarui Iklan"):
							id = []
							post_title = []
							company_name = []
							post_date = []
							job_location = []
							job_des = []
							link = []
	# 						total_employees = []
	# 						actively_recruiting = []

							total_countries = len(countries)				
							if total_countries == 0:
								st.error("Please select at least one country.")
								return
							# countries_string = countries_string.replace(" ", "")	
							if total_countries > jum :  #minimal 1 negara 1 iklan yg dicari. misal 4 negara input dengan pencarian 3 maka muncul warning
								st.warning("The number of countries entered is more than the number of desired ads.")
											
							for i in range(total_countries):
								temp = int(jum/total_countries)
								iterate_number.append(temp)

							sisa = jum%total_countries

							for i in range(sisa):
								iterate_number[i] += 1

							for i in range(len(iterate_number)):

								try:
									# Change root logger level (default is WARN)
									logging.basicConfig(level=logging.INFO)	

									def on_data(data: EventData):

	# 									if selected_filter_actively_recruiting == None:
											post_title.append(
												translator.translate(
													data.title, lang_src="auto", lang_tgt="en"
												)
											)
											# 								post_title.append(data.title)
											id_job = len(post_title)
											id.append(id_job)
											job_location.append(data.place)
											company_name.append(
												translator.translate(
													data.company, lang_src="auto", lang_tgt="en"
												)
											)
											# 								company_name.append(data.company)
											post_date.append(data.date)
											job_des.append(
												translator.translate(
													data.description, lang_src="auto", lang_tgt="en"
												)
											)
											# 								job_des.append(data.description)
											link.append(data.link)
	# 										total_employees.append(data.total_employees)
	# 										actively_recruiting.append(data.actively_recruiting)

	# 									elif data.actively_recruiting == selected_filter_actively_recruiting:
	# 										post_title.append(
	# 											translator.translate(
	# 												data.title, lang_src="auto", lang_tgt="en"
	# 											)
	# 										)
	# 										# 								post_title.append(data.title)
	# 										id_job = len(post_title)
	# 										id.append(id_job)
	# 										job_location.append(data.place)
	# 										company_name.append(
	# 											translator.translate(
	# 												data.company, lang_src="auto", lang_tgt="en"
	# 											)
	# 										)
	# 										# 								company_name.append(data.company)
	# 										post_date.append(data.date)
	# 										job_des.append(
	# 											translator.translate(
	# 												data.description, lang_src="auto", lang_tgt="en"
	# 											)
	# 										)
	# 										# 								job_des.append(data.description)
	# 										link.append(data.link)
	# # 										total_employees.append(data.total_employees)
	# # 										actively_recruiting.append(data.actively_recruiting)

									def on_error(error):
										print("[ON_ERROR]", error)

									def on_end():
										print("[ON_END]")

									scraper = LinkedinScraper(
										# Custom Chrome executable path (e.g. /foo/bar/bin/chromedriver)
										chrome_executable_path="chromedriver",
										chrome_options=None,  # Custom Chrome options here
										headless=True,  # Overrides headless mode only if chrome_options is None
										# How many threads will be spawned to run queries concurrently (one Chrome driver for each thread)
										max_workers=1,
										# Slow down the scraper to avoid 'Too many requests (429)' errors
										slow_mo=1,
									)

									# Add event listeners
									scraper.on(Events.DATA, on_data)
									scraper.on(Events.ERROR, on_error)
									scraper.on(Events.END, on_end)

									queries = [
										Query(
											query=job_title,
											options=QueryOptions(
												locations=countries[i],
												optimize=False,
												limit=int(iterate_number[i]),
												filters=QueryFilters(										
													relevance=RelevanceFilters.RECENT,
													type=jobtype,
													time=time_iklan,
												),
											),
										)
									]

									scraper.run(queries)

								except:
									results = "Not Found"

							job_data = pd.DataFrame(
								{
									"Job_ID": id,
									"Date": post_date,
									"Company Name": company_name,							
	# 								"Total Employees": total_employees,
	# 								"Actively Recruiting": actively_recruiting,
									"Job_Title": post_title,
									"Location": job_location,
									"Description": job_des,
									"Link": link,
								}
							)

							# cleaning description column
							job_data["Description"] = job_data[
								"Description"
							].str.replace("\n", " ")

							# print(job_data.info())
							st.subheader("Data Hasil Scrap")
							# job_data.head()
							job_data.to_csv(
								"datascraptest.csv", index=0, encoding="utf-8"
							)
							dframe = load_data("datascraptest.csv")
							st.dataframe(dframe.head(10))
					else:
						st.error("Please update the ad first.")

					st.title("Download All Result")
					if isSearched == True:
						st.write("Click the button to download ad data")
						# st.write('https://drive.google.com/file/d/1LUyxJgdXEQdPMTuqSCdNjdIKqVjE7z80/view?usp=sharing')
						with open("datascraptest.csv", "rb") as file:
							btn = st.download_button(
								label="Download",
								data=file,
								file_name="data.csv",
								mime="text/csv",
							)	
		elif st.session_state.isUser == True:
			st.warning("You aren't an admin.")
		else:
			st.info(
				"This page can only be accessed by admin. Please login as admin first.")
		
if __name__ == "__main__":
	main()