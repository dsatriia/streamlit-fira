# Core Pkg
from google_trans_new import google_translator
import sqlite3
import hashlib
from webdriver_manager.firefox import GeckoDriverManager
from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.chrome.options import Options
from PIL import Image
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import LsiModel
from gensim import corpora, similarities
import numpy as np
import re
import pandas as pd
import nltk
from webdriver_manager.driver import GeckoDriver
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
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
import gensim
import streamlit as st
import os

os.environ["LI_AT_COOKIE"] = "AQEDATL5DxIAV9iyAAABfXYyjjgAAAF9mj8SOFYApeRK1b5Jn2eD4jwNYUzlMLwNrBBoayd8yMP0B1afCK_HmMyHqCSdjwH3AF8UDMVC3W3pacGjIpIj-Y4vgph6hx3Uis9Cv4mzkSdzncKvgHe3xJLl"
print("debug nih")
print(os.environ['LI_AT_COOKIE'])
# Load LSA

# Packages Scraping

translator = google_translator()

# Packages Pra-Proses

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")

stop_words = set(stopwords.words("english"))
# print(stop_words)
if 'isSearched' not in st.session_state:
    st.session_state['isSearched'] = False

# package gambar


options = Options()


def get_chromedriver_path():
	results = glob.glob(
		"/**/chromedriver", recursive=True
	)  # workaround on streamlit sharing
	which = results[0]
	return which


# packages Link

# Load Our Dataset

def load_data(data):
	df = pd.read_csv(data)
	return df


def local_css(file_name):
	with open(file_name) as f:
		st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Security
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

	menu = ["Home", "Login", "SignUp", "About", "Profiles"]
	choice = st.sidebar.selectbox("Menu", menu)

	isSearched = st.session_state.isSearched
	print("dimaaaaa")
	
	if choice == "Home":
		st.title("Home")
		image = Image.open("logoku.png")
		st.image(image, width=600)

	elif choice == "Login":
		st.title("Login Section")

		username = st.sidebar.text_input("User Name")
		password = st.sidebar.text_input("Password", type="password")
		
		if st.sidebar.checkbox("Login"):
			create_usertable()
			hashed_pswd = make_hashes(password)
			result = login_user(username, check_hashes(password, hashed_pswd))
			if result:
				st.success("Logged In as {}".format(username))
				task = st.selectbox(
					"Task", ["Recommend", "TemplateCV", "Profiles", "Download Data Hasil Scrap"])

				if task == "Recommend":
					st.title("Job Recommender")
					st.subheader("Perbarui Iklan Linkedin")
					Negara = st.text_input("Input Negara")
					job_title = st.text_input("Input Job Title")
					# jum = st.number_input("Input Banyak Iklan yang ingin Ditelusuri")
					jum = st.number_input(
						"Input Banyak Iklan yang ingin Ditelusuri", 2, 100, 5
					)  # mulai,max,default
					if st.button("Perbarui"):
						try:
							# Change root logger level (default is WARN)
							logging.basicConfig(level=logging.INFO)
							id = []
							post_title = []
							company_name = []
							post_date = []
							job_location = []
							job_des = []
							link = []
							total_employees = []
							actively_recruiting = []

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
								total_employees.append(data.total_employees)
								actively_recruiting.append(data.actively_recruiting)
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
										# locations=['Indonesia'],
										locations=Negara,
										optimize=False,
										limit=int(jum),
										filters=QueryFilters(
											#                 company_jobs_url='https://www.linkedin.com/jobs/search/?f_C=1441%2C17876832%2C791962%2C2374003%2C18950635%2C16140%2C10440912&geoId=92000000',  # Filter by companies
											relevance=RelevanceFilters.RECENT,
											#                 type=[TypeFilters.FULL_TIME, TypeFilters.INTERNSHIP],
											#                 experience=None,
										),
									),
								)
							]

							scraper.run(queries)

							job_data = pd.DataFrame(
								{
									"Job_ID": id,
									"Date": post_date,
									"Company Name": company_name,									
									"Total Employees": total_employees,
									"Actively Recruiting": actively_recruiting,
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

						except:
							results = "Not Found"

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

						if st.button("Perbarui Iklan"):
							try:
								# Change root logger level (default is WARN)
								logging.basicConfig(level=logging.INFO)
								id = []
								post_title = []
								company_name = []
								post_date = []
								job_location = []
								job_des = []
								link = []
								total_employees = []
								actively_recruiting = []

								def on_data(data: EventData):
									#     print('[ON_DATA]', data.title, data.company, data.date, data.description, data.link, len(data.description))
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
									total_employees.append(data.total_employees)
									actively_recruiting.append(data.actively_recruiting)

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
											# locations=['Indonesia'],
											locations=Negara,
											optimize=False,
											limit=int(jum),
											filters=QueryFilters(
												#                 company_jobs_url='https://www.linkedin.com/jobs/search/?f_C=1441%2C17876832%2C791962%2C2374003%2C18950635%2C16140%2C10440912&geoId=92000000',  # Filter by companies
												relevance=RelevanceFilters.RECENT,
												type=jobtype,
												time=time_iklan,
												#                 type=[TypeFilters.FULL_TIME, TypeFilters.INTERNSHIP],
												#                 experience=None,
											),
										),
									)
								]

								scraper.run(queries)

								job_data = pd.DataFrame(
									{
										"Job_ID": id,
										"Date": post_date,
										"Company Name": company_name,										
										"Total Employees": total_employees,
										"Actively Recruiting": actively_recruiting,
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

							except:
								results = "Not Found"
					else:
						st.error("Silahkan perbarui iklan terlebih dahulu.")
						
					st.subheader(
						"Upload CV untuk Memukan Rekomendasi Iklan Pekerjaan")
					# st.write(st.session_state.isSearched)
					if isSearched == True:

						file = st.file_uploader("", type="csv")
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
									file, error_bad_lines=False)
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

								lsi_model = LsiModel(
									corpus=corpus_tfidf, id2word=dictionary, num_topics=3
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

								cst = cosine_similarities_test

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
										"Post Date :", f"{documents_train['Date'][i]}\n"
									)
									st.write(
										"Nama Perusahaan :",
										f"{documents_train['Company Name'][i]}\n",
									)
									st.write(
										"Nama Pekerjaan :",
										f"{documents_train['Job_Title'][i]}\n",
									)
									st.write(
										"Deskripsi Pekerjaan :",
										f"{documents_train['Description'][i]}\n",
									)
									st.write(
										"Negara :", f"{documents_train['Location'][i]}\n"
									)
									st.write(
										"Link Iklan pekerjaan :",
										f"{documents_train['Link'][i]}\n",
									)
									st.write(
										"Cosine similiarity cv terhadap iklan:", f"{cst[i]}"
									)

									st.subheader(
										"-----------------------------------------------------------------------------------"
									)

							except:
								results = "Not Found"
					else:
						st.error("Silahkan perbarui iklan terlebih dahulu.")

				elif task == "TemplateCV":
					st.subheader("Download Template CV ")
					st.write("Klik button untuk melakukan download")
					# st.write('https://drive.google.com/file/d/1LUyxJgdXEQdPMTuqSCdNjdIKqVjE7z80/view?usp=sharing')
					with open("templatecv.csv", "rb") as file:
						btn = st.download_button(
							label="Download",
							data=file,
							file_name="template.csv",
							mime="text/csv",
						)
					st.subheader("Panduan Penulisan CV sesuai Template")
					image = Image.open("panduanedit.png")
					st.image(image, caption="Format CV")

				elif task == "Profiles":
					st.subheader("User Profiles")
					user_result = view_all_users()
					clean_db = pd.DataFrame(
						user_result, columns=["Username", "Password", "Role"]
					)
					st.dataframe(clean_db)

 # START
				elif task == "Download Data Hasil Scrap":
					st.subheader("Download Data scrap ")
					st.write("Klik button untuk melakukan download data iklan")
					# st.write('https://drive.google.com/file/d/1LUyxJgdXEQdPMTuqSCdNjdIKqVjE7z80/view?usp=sharing')
					with open("datascraptest.csv", "rb") as file:
						btn = st.download_button(
							label="Download",
							data=file,
							file_name="data.csv",
							mime="text/csv",
						)
# END
		else:
			st.warning("Incorrect Username/Password")

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
		st.write("Dapatkan rekomendasi pekerjaan yang sesuai dengan curriculum vitae")

	elif choice == "Profiles":
		username = st.sidebar.text_input("Username Admin")
		password = st.sidebar.text_input("Password Admin", type="password")

		st.title("Profiles")

		if st.sidebar.checkbox("Login Admin"):
			create_usertable()
			hashed_pswd = make_hashes(password)
			result_admin = login_admin(
				username, check_hashes(password, hashed_pswd))
			result_user = login_user(
				username, check_hashes(password, hashed_pswd))
			if result_admin:
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
			elif result_user:
				st.warning("Anda hanya user biasa, bukan admin.")
			else:
				st.error("Incorrect Username/Password")
		else:
			st.info(
				"Halaman ini hanya bisa diakses oleh admin. Silahkan login sebagai admin terlebih dahulu.")


if __name__ == "__main__":
	main()
