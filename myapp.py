# Core Pkg
import streamlit as st 
import os


# Load LSA
import gensim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity,linear_kernel

#Packages Scraping
import logging
from linkedin_jobs_scraper import LinkedinScraper
from linkedin_jobs_scraper.events import Events, EventData
from linkedin_jobs_scraper.query import Query, QueryOptions, QueryFilters
from linkedin_jobs_scraper.filters import RelevanceFilters, TimeFilters, TypeFilters, ExperienceLevelFilters
from google_trans_new import google_translator  
translator = google_translator() 

#Packages Pra-Proses
import nltk
import pandas as pd
import re
import numpy as np
from gensim import corpora, similarities
from gensim.models import LsiModel
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
stop_words = set(stopwords.words('english'))
print (stop_words)

from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.wait import WebDriverWait

options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")
options.add_argument("--disable-features=NetworkService")
options.add_argument("--window-size=1920x1080")
options.add_argument("--disable-features=VizDisplayCompositor")

def get_chromedriver_path():
    results = glob.glob('/**/chromedriver', recursive=True)  # workaround on streamlit sharing
    which = results[0]
    return which
	
#packages Link
from selenium import webdriver

# Load Our Dataset
def load_data(data):
	df = pd.read_csv(data)
	return df 

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def main():
	local_css("style.css")

	st.image("kotaklogo.png")
	st.title("Match CV to Jobs App")


	menu = ["Home","Login","Recommend","About"]
	choice = st.sidebar.selectbox("Menu",menu)

	df = load_data("datascraptest2.csv")

	if choice == "Home":
		st.subheader("Kunjungi Menu Recommend di Sidebar untuk Menemukan Rekomendasi Pekerjaan")
		def file_selector(folder_path='.'):
			filenames = os.listdir(folder_path)
			selected_filename = st.selectbox('Select a file', filenames)
			return os.path.join(folder_path, selected_filename)

		
		st.write('Download Template CV disini')
				
		if st.button("Download_CV"):
			try:
				driver = webdriver.Chrome (options=options, service_log_path='selenium.log')
				# driver.get("https://drive.google.com")
				# identify element with partial link text and click()
				#l=driver.find_element_by_partial_link_text("Download")
				#l.click()
			except:
					results= "Not Found"
		
	elif choice == "Login":
		st.subheader("Login Section")
		username = st.text_input("User Name")
		password = st.text_input("Password", type='password')
		if st.button("Login"):
			if (username == 'fira' and password == 'fira123'):
				st.success("Logged In as {}".format(username))

			else:
				st.warning("Incorrect Username/Password")


	elif choice == "Recommend":
		st.subheader("Recommend Job")
		st.subheader("Data Iklan")
		st.dataframe(df.head(10))
		st.subheader("Perbarui Iklan Linkedin")
		Negara = st.text_input("Input Negara")		
		job_title = st.text_input("Input Job Title")
		#jum = st.number_input("Input Banyak Iklan yang ingin Ditelusuri")
		filter_time = [TimeFilters.DAY,TimeFilters.WEEK,TimeFilters.MONTH,TimeFilters.ANY]
		time_iklan = st.selectbox("Date Posted",filter_time)
		jum = st.number_input("Input Banyak Iklan yang ingin Ditelusuri",2,100,5)#mulai,max,default
		if st.button("Perbarui"):
			try:
				# Change root logger level (default is WARN)
				logging.basicConfig(level = logging.INFO)
				id = []
				post_title = []
				company_name = []
				post_date = []
				job_location = []
				job_des = []
				link = []

				def on_data(data: EventData):
				#     print('[ON_DATA]', data.title, data.company, data.date, data.description, data.link, len(data.description))
					post_title.append(translator.translate(data.title, lang_src='auto',lang_tgt='en'))
					id_job = (len(post_title))
					id.append(id_job)
					job_location.append(data.place)
					company_name.append(translator.translate(data.company, lang_src='auto',lang_tgt='en'))
					post_date.append(data.date)
					#job_desc = translator.translate(data.description, lang_src='auto',lang_tgt='en')
					job_des.append(translator.translate(data.description, lang_src='auto',lang_tgt='en'))
					link.append(data.link)
					
				def on_error(error):
					print('[ON_ERROR]', error)

				def on_end():
					print('[ON_END]')

				scraper = LinkedinScraper(
					chrome_executable_path=None, # Custom Chrome executable path (e.g. /foo/bar/bin/chromedriver) 
					chrome_options=None,  # Custom Chrome options here
					headless=True,  # Overrides headless mode only if chrome_options is None
					max_workers=1,  # How many threads will be spawned to run queries concurrently (one Chrome driver for each thread)
					slow_mo=1,  # Slow down the scraper to avoid 'Too many requests (429)' errors
				)

				# Add event listeners
				scraper.on(Events.DATA, on_data)
				scraper.on(Events.ERROR, on_error)
				scraper.on(Events.END, on_end)

				queries = [
					Query(
						query=job_title,
						options=QueryOptions(
							#locations=['Indonesia'],
							locations=Negara,
							optimize=False,
							limit=int(jum),
							filters=QueryFilters(
				#                 company_jobs_url='https://www.linkedin.com/jobs/search/?f_C=1441%2C17876832%2C791962%2C2374003%2C18950635%2C16140%2C10440912&geoId=92000000',  # Filter by companies
				                 relevance=RelevanceFilters.RECENT,
				                 time=time_iklan,
				#                 type=[TypeFilters.FULL_TIME, TypeFilters.INTERNSHIP],
				#                 experience=None,
							)
						)
					)
				]

				scraper.run(queries)	
				
				job_data = pd.DataFrame({'Job_ID':id,
				'Date': post_date,
				'Company Name': company_name,
				'Post': post_title,
				'Location': job_location,
				'Description': job_des,
				'Link': link,
										 
				})

				# cleaning description column
				job_data['Description'] = job_data['Description'].str.replace('\n',' ')
					
				
				print(job_data.info())
				st.subheader("Data Hasil Scrap")
				#job_data.head()
				job_data.to_csv('datascraptest2.csv', index=0, encoding='utf-8')
				dframe = load_data("datascraptest2.csv")
				st.dataframe(dframe.head(10))
				
			except:
				results= "Not Found"
				
		st.subheader("Upload CV untuk Memukan Rekomendasi Iklan Pekerjaan")
		file = st.file_uploader("", type='csv')
		jumlah = st.number_input("Input Banyak Iklan yang ingin Ditampilkan",2,100,10)#mulai,max,default
		if st.button("Temukan Iklan yang Cocok"):
			try:
				def remove_punc(text):
					symbols = r"â€¢!\"#$%&()*+-.,/:;<=>?@[\]^_`'{|}~1234567890\\" 
					output = [char for char in text if char not in symbols]
					return "".join(output)

				def stemSentence(tokens):
				#token_words=word_tokenize(text)
					stem_sentence=[]
					for word in tokens:
						stem_sentence.append(porter.stem(word))
					return stem_sentence

				def lemmatization(tokens):
					lemmatizer = WordNetLemmatizer() 
					return [lemmatizer.lemmatize(word) for word in tokens ]

				def stopwordSentence(tokens):
					return [word for word in tokens if word not in stop_words]

				def caseFold(text):
					return text.lower()

				def preProcessPipeline(text, print_output=False):
					if print_output:
						print('Teks awal:')
						print(text)
					text = remove_punc(text)
					if print_output:
						print('Setelah menghilangkan tanda baca:')
						print(text)

					text = caseFold(text)
					if print_output:
						print('Setelah Casefold')
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
				documents_train = pd.read_csv('datascraptest2.csv', error_bad_lines=False)
				train_text = documents_train['Description'].apply(preProcessPipeline)
				documents_test = pd.read_csv(file, error_bad_lines=False)
				test_text = documents_test['cv_desc'].apply(preProcessPipeline)

				nltk_tokens = [nltk.word_tokenize(i) for i in train_text]
				y = nltk_tokens

				dictionary = gensim.corpora.Dictionary(y)
				text = y

				corpus = [dictionary.doc2bow(i) for i in text]
				tfidf = gensim.models.TfidfModel(corpus, smartirs='npu')
				corpus_tfidf = tfidf[corpus]

				lsi_model = LsiModel(corpus=corpus_tfidf,id2word=dictionary, num_topics = 3)
				print("Derivation of Term Matrix T of Training Document Word Stems: ",lsi_model.get_topics())
					#Derivation of Term Document Matrix of Training Document Word Stems = M' x [Derivation of T]
				print("LSI Vectors of Training Document Word Stems: ",[lsi_model[document_word_stems] for document_word_stems in corpus])
				cosine_similarity_matrix = similarities.MatrixSimilarity(lsi_model[corpus])


				word_tokenizer = nltk.tokenize.WordPunctTokenizer()
				words = word_tokenizer.tokenize(test_text[0])

				vector_lsi_test = lsi_model[dictionary.doc2bow(words)]

				cosine_similarities_test = cosine_similarity_matrix[vector_lsi_test]

				most_similar_document_test = train_text[np.argmax(cosine_similarities_test)]

				cst = (cosine_similarities_test)
				
				cst_terurut = sorted(cosine_similarities_test, reverse=True)

				iklan = cosine_similarities_test.argsort()[-jumlah:][::-1]
				def generator_cosines(iklan):
					for i in iklan:
						yield i						
				#print data awal di csv 
				for i in iklan:	
						st.write("Post Date :",f"{documents_train['Date'][i]}\n" )
						st.write("Nama Perusahaan :",f"{documents_train['Company Name'][i]}\n" )
						st.write("Nama Pekerjaan :",f"{documents_train['Post'][i]}\n" )
						st.write("Deskripsi Pekerjaan :",f"{documents_train['Description'][i]}\n" )
						st.write("Negara :",f"{documents_train['Location'][i]}\n" )
						st.write("Link Iklan pekerjaan :",f"{documents_train['Link'][i]}\n" )
						st.write("Cosine similiarity cv terhadap iklan:", f"{cst[i]}")
						st.subheader("-----------------------------------------------------------------------------------")
						
				
			except:
				results= "Not Found"
		
	else:
		st.subheader("About")
		st.text("Built with Streamlit & Pandas")


if __name__ == '__main__':
	main()