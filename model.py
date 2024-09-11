import os
import requests
import pdfplumber
import pytesseract
from PIL import Image
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
import google.generativeai as genai


class Gemini:
    def __init__(self, pdf_url, dictionary, other):
        load_dotenv()
        self.API = os.getenv('GEMINI_API_KEY')
        genai.configure(api_key=self.API)
        self.dictionary = dictionary
        self.get_result(pdf_url, other)

    def download_pdf(self, url, local_filename):
        response = requests.get(url)
        response.raise_for_status()
        with open(local_filename, 'wb') as f:
            f.write(response.content)

    def extract_data_from_pdf(self, pdf_path):
        consolidated_data = ""

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    consolidated_data += f"=== Text from Page {page.page_number} ===\n"
                    consolidated_data += page_text.strip() + "\n\n"

                tables = self.extract_tables_from_page(page)
                if tables:
                    consolidated_data += f"=== Tables from Page {page.page_number} ===\n"
                    for table in tables:
                        consolidated_data += "\n".join(table['rows']) + "\n\n"

                images = page.images
                if images:
                    consolidated_data += f"=== Images from Page {page.page_number} ===\n"
                    for i, img in enumerate(images, start=1):
                        img_data = self.extract_text_from_image(img['stream'])
                        consolidated_data += f"Image {i} OCR Result:\n{img_data}\n\n"

        return consolidated_data.strip()

    def extract_tables_from_page(self, page):
        tables = []
        lines = page.extract_text().split('\n')

        for line in lines:
            if len(line.split('\t')) > 1:
                columns = line.split('\t')
                table_data = {
                    'rows': [line],
                    'columns': [columns]
                }
                tables.append(table_data)

        return tables

    def extract_text_from_image(self, image_stream):
        try:
            img = Image.open(image_stream)
            img_text = pytesseract.image_to_string(img)
            return img_text
        except Exception as e:
            return ""

    def pdf_text(self, pdf_path):
        return self.extract_data_from_pdf(pdf_path)

    def get_chunks(self, text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        return text_splitter.split_text(text)

    def get_vectors(self, text_chunks):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=self.API)
        vector = FAISS.from_texts(text_chunks, self.embeddings)
        vector.save_local("faiss_index")

    def get_conversational_chain(self):
        prompt_template = """
        Please provide the following information strictly in the format below:
        `Answer concisely and informatively, aiming to provide relevant values even if the exact key is not found. Use semantic understanding to match keys and values where possible. Avoid returning "Not Available" unless absolutely necessary. Ensure no key is left empty and present responses/output strictly in the format key without any additional characters.`
        
        Context: \n{context}?\n
        Question: \n{question}\n

        Answer:
        """
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
            google_api_key=self.API
        )
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain

    def user_input(self, question):
        db = FAISS.load_local("faiss_index", self.embeddings, allow_dangerous_deserialization=True)
        docs = db.similarity_search(question)
        
        chain = self.get_conversational_chain()
        response = chain.invoke(
            {"input_documents": docs, "question": question, "return_only_outputs": True}
        )
        
        return response["output_text"]

    def get_result(self, pdf_url, other):
        other.update_log.emit("Downloading PDF from URL...")
        pdf_path = "downloaded_file.pdf"
        try:
            self.download_pdf(pdf_url, pdf_path)
            other.update_log.emit("Extracting Text and Tables from PDF...")
            text_content = self.pdf_text(pdf_path)
            other.update_log.emit("Making Text Chunks...")
            text_chunks = self.get_chunks(text_content)
            other.update_log.emit("Creating Vector Embeddings...")
            self.get_vectors(text_chunks)
            other.update_log.emit("Collecting Response...")
            
            question = """
                    """
            
            response_text = self.user_input(question)
            print('Response text:\n', response_text)
            self.update_data(response_text)
            other.update_log.emit("Data extracted successfully.")
        except Exception as e:
            other.update_log.emit(f"Error during processing: {str(e)}")
            self.dictionary.update({
                'NOI': 'Not Available',
                'Pro-Forma NOI': 'Not Available',
                'Rental': 'Not Available',
                'Total Expense': 'Not Available',
                'Total Revenue': 'Not Available',
                'Economic Occupancy': 'Not Available',
                'Unit Occupancy': 'Not Available',
                'SQFT Occupancy': 'Not Available'
            })
        
        try:
            os.remove(pdf_path)
        except:
            pass

    def update_data(self, response_text):
        try:
            try:
                lines = response_text.replace('```', '').replace('json', '').split('\n')
                lines = lines[1:-1]
            except:
                lines = response_text.strip().split('\n')
            for data in lines:
                if data:
                    key, value = map(str.strip, data.split(':', 1))
                    if value and value.lower() not in ['null', 'none', 'not available']:
                        self.dictionary[key] = value
                        print(f'{key}: {value}')
                    else:
                        self.dictionary[key] = 'Not Available'
        except Exception as e:
            print(f"Error updating data: {e}")
            self.dictionary.update({
                'NOI': 'Not Available',
                'Pro-Forma NOI': 'Not Available',
                'Rental': 'Not Available',
                'Total Expense': 'Not Available',
                'Total Revenue': 'Not Available',
                'Economic Occupancy': 'Not Available',
                'Unit Occupancy': 'Not Available',
                'SQFT Occupancy': 'Not Available'
            })


if __name__ == "__main__":
    gemini = Gemini('<PDF.pdf>', {}, '')
    
    print(gemini.dictionary)
    
