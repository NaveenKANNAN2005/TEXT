import fitz  
import easyocr
import cv2
import os
from docx import Document
from transformers import pipeline, AutoTokenizer
import matplotlib.pyplot as plt
from deep_translator import GoogleTranslator
import time
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import sys
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import math
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
from typing import List, Optional
import uvicorn
from pydantic import BaseModel
import base64
import io
from PIL import Image
import numpy as np
from langdetect import detect
from textblob import TextBlob
from collections import Counter
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="OCR API", description="API for OCR, translation, and text summarization")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create upload directory if it doesn't exist
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
logger.info(f"Upload directory: {UPLOAD_DIR}")

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    logger.error(f"Failed to download NLTK data: {e}")

# Initialize the summarization pipeline
try:
    summarizer = pipeline("summarization", framework="pt")
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    logger.info("Summarization pipeline initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize summarization pipeline: {e}")
    summarizer = None
    tokenizer = None

# Constants for chunk sizes and buffer limits
MAX_CHUNK_SIZE = 1000  # Maximum number of characters per chunk for translation
MAX_DISPLAY_LENGTH = 1000  # Maximum number of characters to display in console
PROCESSING_CHUNK_SIZE = 5000  # Number of characters to process at once for large files

# Pydantic models for request/response
class SummarizationOptions(BaseModel):
    line_count: int = 3
    max_length: Optional[int] = None
    min_length: Optional[int] = None
    do_sample: bool = False

class ProcessingResponse(BaseModel):
    status: str
    text: str
    summary: Optional[str] = None
    translated_text: Optional[str] = None
    confidence: Optional[float] = None
    analysis: Optional[dict] = None

class RealTimeDisplay:
    def __init__(self):
        self.text_queue = queue.Queue()
        self.display_thread = None
        self.running = False
        
    def start(self):
        self.running = True
        self.display_thread = threading.Thread(target=self._display_loop)
        self.display_thread.daemon = True
        self.display_thread.start()
        
    def stop(self):
        self.running = False
        if self.display_thread:
            self.display_thread.join()
            
    def add_text(self, text):
        self.text_queue.put(text)
        
    def _display_loop(self):
        while self.running:
            try:
                text = self.text_queue.get(timeout=0.1)
                # Clear the last line
                sys.stdout.write('\r' + ' ' * 80 + '\r')
                # Display the text with truncation if needed
                display_text = text[:MAX_DISPLAY_LENGTH] + "..." if len(text) > MAX_DISPLAY_LENGTH else text
                sys.stdout.write(display_text + '\n')
                sys.stdout.flush()
            except queue.Empty:
                continue

def translate_text(text, target_language):
    """Translate text to the target language using Google Translator with chunking for large texts."""
    try:
        if not text.strip():
            return text

        # Split text into smaller chunks to avoid translation API limits
        chunks = [text[i:i + MAX_CHUNK_SIZE] for i in range(0, len(text), MAX_CHUNK_SIZE)]
        translated_chunks = []

        print("\nTranslating text chunks...")
        translator = GoogleTranslator(source='auto', target=target_language)
        
        for chunk in tqdm(chunks, desc="Translation Progress"):
            try:
                translated = translator.translate(chunk)
                translated_chunks.append(translated)
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                print(f"\nWarning: Translation error for chunk: {e}")
                translated_chunks.append(chunk)  # Keep original on error

        return ' '.join(translated_chunks)
    except Exception as e:
        print(f"\nError during translation: {e}")
        return text

def extract_text_from_pdf(pdf_path):
    """Extract text directly from a PDF file. Uses OCR for scanned pages."""
    try:
        print("\nProcessing PDF...")
        pdf_document = fitz.open(pdf_path)
        text_content = []
        total_pages = len(pdf_document)
        reader = None  # Initialize OCR reader only if needed

        for page_num in range(total_pages):
            try:
                page = pdf_document[page_num]
                text = page.get_text()
                
                # If no text is found, the page might be scanned
                if not text.strip():
                    if reader is None:
                        print("Initializing OCR for scanned page...")
                        reader = easyocr.Reader(['en'])  # Initialize with English
                    
                    # Convert page to image
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    img_np = np.array(img)
                    
                    # Perform OCR
                    results = reader.readtext(img_np)
                    text = "\n".join([result[1] for result in results])
                    print(f"OCR performed on page {page_num + 1}")
                
                text_content.append(f"--- Page {page_num + 1} ---\n{text}")
                print(f"Processed page {page_num + 1}/{total_pages}")
            except Exception as e:
                print(f"Error processing page {page_num + 1}: {e}")
                text_content.append(f"--- Page {page_num + 1} ---\n[Error: Could not extract text]")

        pdf_document.close()
        return "\n".join(text_content)
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_image(image_path, languages=['en']):
    """Extract text from an image file using EasyOCR."""
    try:
        print("Processing image...")
        reader = easyocr.Reader(languages, gpu=False)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Failed to read image. Please check the file path.")
        results = reader.readtext(image)
        extracted_text = ""
        for (bbox, text, prob) in results:
            extracted_text += f"{text} (Confidence: {prob:.2f})\n"
        return extracted_text, results
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return "", []

def extract_text_from_word(docx_path):
    """Extract text from a Word document."""
    try:
        print("Processing Word document...")
        document = Document(docx_path)
        all_text = ""
        for paragraph in document.paragraphs:
            all_text += paragraph.text + "\n"
        return all_text
    except Exception as e:
        print(f"Error extracting text from Word document: {e}")
        return ""

def chunk_text(text, max_tokens=1024):
    """Split text into chunks that fit within the model's token limit."""
    try:
        if not tokenizer:
            logger.error("Tokenizer not initialized")
            raise ValueError("Tokenizer not available")

        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Try adding the sentence to the current chunk
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            input_ids = tokenizer.encode(test_chunk, add_special_tokens=False)
            
            if len(input_ids) > max_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # If the sentence itself is too long, split it by tokens
                if len(tokenizer.encode(sentence, add_special_tokens=False)) > max_tokens:
                    words = sentence.split()
                    current_chunk = words[0]
                    for word in words[1:]:
                        test_chunk = current_chunk + " " + word
                        if len(tokenizer.encode(test_chunk, add_special_tokens=False)) > max_tokens:
                            chunks.append(current_chunk.strip())
                            current_chunk = word
                        else:
                            current_chunk = test_chunk
                else:
                    current_chunk = sentence
            else:
                current_chunk = test_chunk

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    except Exception as e:
        logger.error(f"Error in chunk_text: {e}")
        raise ValueError(f"Failed to chunk text: {str(e)}")

def summarize_text(text, line_count=3, advanced_options=None):
    """Summarize text using the transformers pipeline."""
    try:
        if not text or len(text.strip()) == 0:
            logger.error("Empty text provided for summarization")
            return "No text to summarize."

        if not summarizer:
            logger.error("Summarizer not initialized")
            raise ValueError("Summarization service not available")

        # Default options
        max_length = 150
        min_length = 50
        do_sample = False

        # Update with advanced options if provided
        if advanced_options:
            max_length = advanced_options.get('max_length', max_length)
            min_length = advanced_options.get('min_length', min_length)
            do_sample = advanced_options.get('do_sample', do_sample)

        # Ensure text is long enough to summarize
        if len(text.split()) < min_length:
            logger.warning("Text too short for summarization")
            return text

        # Split text into chunks that fit within the model's token limit
        chunks = chunk_text(text)
        summaries = []

        logger.info(f"Summarizing text with {len(chunks)} chunks")
        for chunk in chunks:
            try:
                summary = summarizer(
                    chunk,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=do_sample
                )[0]['summary_text']
                summaries.append(summary)
            except Exception as e:
                logger.error(f"Error summarizing chunk: {e}")
                continue

        if not summaries:
            logger.error("No summaries generated")
            return "Failed to generate summary."

        # Combine summaries
        final_summary = " ".join(summaries)
        
        # Split into sentences and take the first 'line_count' sentences
        sentences = sent_tokenize(final_summary)
        selected_sentences = sentences[:line_count]
        
        return " ".join(selected_sentences)

    except Exception as e:
        logger.error(f"Error in summarize_text: {e}")
        raise ValueError(f"Failed to summarize text: {str(e)}")

def display_image_with_text(image_path, results):
    """Display the image with detected text boxes and annotations using matplotlib."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Failed to read image. Please check the file path.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for (bbox, text, prob) in results:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))

            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(image, f"{text} ({prob:.2f})", (top_left[0], top_left[1] - 10), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        plt.figure(figsize=(12, 12))
        plt.imshow(image)
        plt.axis('off')
        plt.title("Detected Text with Annotations")
        plt.show()
    except Exception as e:
        print(f"Error displaying image: {e}")

def save_to_file(output_text, output_path):
    """Save the extracted or summarized text to a file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(output_text)
        print(f"\nText saved to {output_path}")
    except Exception as e:
        print(f"Error saving text to file: {e}")

def is_supported_image(filename):
    """Check if the file is a supported image type (case-insensitive)."""
    supported_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    return filename.lower().endswith(supported_exts)

def process_large_file(file_path, output_dir, languages, target_language):
    """Process large files with real-time display and chunked processing."""
    try:
        # Initialize real-time display
        display = RealTimeDisplay()
        display.start()

        # Create output files
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        extracted_path = os.path.join(output_dir, f"{base_name}_extracted.txt")
        summary_path = os.path.join(output_dir, f"{base_name}_summary.txt")

        # Process file based on type
        if file_path.lower().endswith('.pdf'):
            text_generator = extract_text_from_pdf(file_path)
        else:
            print("Currently only PDF files are supported for large file processing")
            display.stop()
            return

        # Process and save extracted text in chunks
        print("\nProcessing and saving extracted text...")
        with open(extracted_path, 'w', encoding='utf-8') as extracted_file:
            for chunk in text_generator.split('\n'):
                if target_language != "en":
                    chunk = translate_text(chunk, target_language)
                extracted_file.write(chunk + '\n')
                display.add_text(f"Processed chunk of size: {len(chunk)} characters")

        # Ask for summarization
        summarize_choice = input("\nWould you like to summarize the text? (yes/no): ").strip().lower()
        if summarize_choice in ['yes', 'y']:
            try:
                line_count = int(input("How many lines should the summary be? "))
                
                print("\nGenerating summary...")
                with open(extracted_path, 'r', encoding='utf-8') as f:
                    text = f.read()

                # Process summary in chunks
                chunks = chunk_text(text)
                all_summaries = []

                for i, chunk in enumerate(tqdm(chunks, desc="Summarizing")):
                    summary = summarize_text(chunk, line_count=line_count)
                    if summary:
                        all_summaries.append(summary)
                        display.add_text(f"Summarized chunk {i+1}/{len(chunks)}")

                final_summary = " ".join(all_summaries)
                
                # Translate summary if needed
                if target_language != "en":
                    final_summary = translate_text(final_summary, target_language)

                # Save summary
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write(final_summary)
                print(f"\nSummary saved to: {summary_path}")

            except ValueError as e:
                print(f"\nError in summarization: {e}")

        display.stop()
        print(f"\nExtracted text saved to: {extracted_path}")

    except Exception as e:
        print(f"\nError processing large file: {e}")
        if 'display' in locals():
            display.stop()

def process_file(file_path, output_dir, languages, target_language, add_delay=False):
    """Process a single file based on its type."""
    try:
        # Determine the output directory
        if os.path.isabs(file_path):
            # If input path is absolute, create Processed folder in the same directory
            parent_dir = os.path.dirname(file_path)
            output_dir = os.path.join(parent_dir, "Processed")
        else:
            # If input path is relative, use the provided output_dir
            output_dir = os.path.join(os.getcwd(), output_dir)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nOutput directory: {output_dir}")

        # Extract text based on file type
        print(f"\nProcessing file: {file_path}")
        if file_path.lower().endswith('.pdf'):
            extracted_text = extract_text_from_pdf(file_path)
            if not extracted_text:
                print("Error: No text could be extracted from the PDF")
                return
        elif is_supported_image(file_path):
            extracted_text, results = extract_text_from_image(file_path, languages)
            if results:
                display_image_with_text(file_path, results)
        elif file_path.lower().endswith('.docx'):
            extracted_text = extract_text_from_word(file_path)
        else:
            print(f"Skipping unsupported file format: {file_path}")
            return

        # Create base filename for output
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        extracted_path = os.path.join(output_dir, f"{base_name}_extracted.txt")

        # Translate extracted text if needed
        if target_language != "en" and extracted_text.strip():
            print("\nTranslating extracted text...")
            try:
                translated_text = translate_text(extracted_text, target_language)
                if add_delay:
                    time.sleep(2)
            except Exception as e:
                print(f"Translation error: {e}")
                translated_text = extracted_text
        else:
            translated_text = extracted_text

        # Display and save extracted text
        if translated_text.strip():
            print(f"\nExtracted Text from {file_path}:")
            # Display first 1000 characters with page break for readability
            display_text = translated_text[:1000] + "..." if len(translated_text) > 1000 else translated_text
            print(display_text.replace("\n\n", "\n"))  # Reduce double line breaks for cleaner display
            
            try:
                with open(extracted_path, 'w', encoding='utf-8') as f:
                    f.write(translated_text)
                print(f"\nExtracted text saved to: {extracted_path}")
            except Exception as e:
                print(f"Error saving extracted text: {e}")
        else:
            print("No text content to save.")
            return

        # Handle summarization
        summarize_choice = input(f"\nWould you like to summarize the text from {base_name}? (yes/no): ").strip().lower()
        if summarize_choice in ['yes', 'y']:
            try:
                line_count = int(input("How many lines should the summary be? "))
                advanced_choice = input("Would you like to enable advanced options for summarization? (yes/no): ").strip().lower()
                advanced_options = None

                if advanced_choice in ['yes', 'y']:
                    max_length = int(input("Enter max length for the summary: "))
                    min_length = int(input("Enter min length for the summary: "))
                    do_sample = input("Enable sampling (yes/no): ").strip().lower() in ['yes', 'y']
                    advanced_options = {
                        'max_length': max_length,
                        'min_length': min_length,
                        'do_sample': do_sample
                    }

                print("\nSummarizing the text...")
                summarized_text = summarize_text(extracted_text, line_count=line_count, advanced_options=advanced_options)

                if summarized_text and target_language != "en":
                    print("\nTranslating summary...")
                    try:
                        summarized_text = translate_text(summarized_text, target_language)
                        if add_delay:
                            time.sleep(2)
                    except Exception as e:
                        print(f"Summary translation error: {e}")

                if summarized_text:
                    print(f"\nSummarized Text from {file_path}:")
                    print(summarized_text)

                    # Save summary to a separate file
                    summary_path = os.path.join(output_dir, f"{base_name}_summary.txt")
                    try:
                        with open(summary_path, 'w', encoding='utf-8') as f:
                            f.write(summarized_text)
                        print(f"\nSummarized text saved to: {summary_path}")
                    except Exception as e:
                        print(f"Error saving summary: {e}")
                else:
                    print("Could not generate summary.")

            except ValueError as e:
                print(f"Invalid input: {e}")
            except Exception as e:
                print(f"Error during summarization: {e}")

    except Exception as e:
        print(f"An unexpected error occurred while processing {file_path}: {e}")


def process_directory(directory_path, languages, target_language):
    """Process all supported files in a directory."""
    try:
        output_dir = os.path.join(directory_path, "Processed")
        os.makedirs(output_dir, exist_ok=True)

        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                process_file(file_path, output_dir, languages, target_language, add_delay=True)

        print(f"\nAll files processed. Output saved in: {output_dir}")
    except Exception as e:
        print(f"Error processing directory: {e}")


def main():
    # Specify the path to process
    input_path = input("Enter the file or directory path to process: ").strip()

    # Specify OCR languages
    language_input = input("Enter OCR languages (comma-separated, e.g., 'en,ta,hi'): ").strip()
    languages = [lang.strip() for lang in language_input.split(',')]

    # Specify target language for translation
    target_language = input("Enter target language for translation (e.g., 'ta' for Tamil, 'hi' for Hindi, 'en' for English): ").strip()

    # Check if the input is a file or directory
    if os.path.isfile(input_path):
        # Use a dedicated output directory for single file for consistency
        output_dir = os.path.join(os.path.dirname(input_path), "Processed")
        os.makedirs(output_dir, exist_ok=True)
        process_file(input_path, output_dir, languages, target_language)
    elif os.path.isdir(input_path):
        process_directory(input_path, languages, target_language)
    else:
        print(f"Invalid path: {input_path}")

# New API endpoints
@app.post("/api/upload", response_model=ProcessingResponse)
async def upload_file(
    file: UploadFile = File(...),
    languages: str = Form("en"),
    target_language: str = Form("en"),
    summarize: bool = Form(False),
    mode: str = Form("extract"),
    summarization_options: Optional[str] = Form(None)
):
    temp_file_path = None
    try:
        logger.info(f"Processing file: {file.filename}, mode: {mode}")
        
        # Create temporary file path
        temp_file_path = os.path.join(UPLOAD_DIR, file.filename)
        logger.info(f"Temporary file path: {temp_file_path}")
        
        # Save uploaded file
        try:
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info("File saved successfully")
        except Exception as e:
            logger.error(f"Failed to save file: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
        
        # Process based on file type and mode
        file_ext = os.path.splitext(file.filename)[1].lower()
        langs = languages.split(',')
        logger.info(f"File extension: {file_ext}, Languages: {langs}")
        
        # Extract text based on file type
        try:
            if file_ext == '.pdf':
                extracted_text = extract_text_from_pdf(temp_file_path)
                confidence = 1.0
            elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                extracted_text, results = extract_text_from_image(temp_file_path, langs)
                confidence = sum(result[2] for result in results) / len(results) if results else 1.0
            elif file_ext in ['.doc', '.docx']:
                extracted_text = extract_text_from_word(temp_file_path)
                confidence = 1.0
            elif file_ext == '.txt':
                with open(temp_file_path, 'r', encoding='utf-8') as f:
                    extracted_text = f.read()
                confidence = 1.0
            else:
                raise HTTPException(status_code=400, detail="Unsupported file format")
            
            if not extracted_text:
                extracted_text = ""
                logger.warning("No text extracted from file")
            
            logger.info("Text extracted successfully")
        except Exception as e:
            logger.error(f"Failed to extract text: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to extract text: {str(e)}")

        # Initialize response variables
        translated_text = None
        summary = None
        analysis = None

        # Process based on mode
        try:
            if mode == "translate":
                if target_language != "en" and extracted_text:
                    translated_text = translate_text(extracted_text, target_language)
                    logger.info("Translation completed")
            
            elif mode == "summarize" or summarize:
                if not summarizer:
                    raise HTTPException(status_code=500, detail="Summarization service not available")
                
                options = None
                if summarization_options:
                    options = SummarizationOptions.parse_raw(summarization_options)
                    summary = summarize_text(
                        extracted_text,
                        line_count=options.line_count,
                        advanced_options={
                            'max_length': options.max_length,
                            'min_length': options.min_length,
                            'do_sample': options.do_sample
                        }
                    )
                else:
                    summary = summarize_text(extracted_text)
                logger.info("Summarization completed")
                
                # For summarize mode, we don't need to return the original text
                if mode == "summarize":
                    extracted_text = ""

            elif mode == "analysis":
                try:
                    analysis = {
                        "Basic Statistics": {
                            "Document Type": file_ext.upper()[1:] + " Document",
                            "Character Count": len(extracted_text),
                            "Word Count": len(extracted_text.split()),
                            "Line Count": len(extracted_text.splitlines()),
                            "Language": detect_language(extracted_text)
                        },
                        "Key Topics": extract_key_topics(extracted_text),
                        "Sentiment Analysis": analyze_sentiment(extracted_text),
                        "Readability Metrics": calculate_readability(extracted_text),
                        "Document Structure": analyze_text_structure(extracted_text)
                    }
                    logger.info("Analysis completed")
                except Exception as e:
                    logger.error(f"Error during document analysis: {e}")
                    analysis = {
                        "error": f"Failed to complete analysis: {str(e)}"
                    }
            
        except Exception as e:
            logger.error(f"Failed to process in mode {mode}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process in mode {mode}: {str(e)}")
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    logger.info("Temporary file cleaned up")
                except Exception as e:
                    logger.error(f"Error cleaning up temporary file: {e}")

        # Create response with required fields
        return ProcessingResponse(
            status="success",
            text=extracted_text if extracted_text is not None else "",
            translated_text=translated_text,
            summary=summary,
            confidence=confidence,
            analysis=analysis
        )

    except Exception as e:
        logger.error(f"Unexpected error in upload_file: {e}")
        # Ensure cleanup in case of any error
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info("Temporary file cleaned up after error")
            except Exception as cleanup_error:
                logger.error(f"Failed to clean up temporary file after error: {cleanup_error}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process-image-base64")
async def process_image_base64(
    image_data: str = Form(...),
    languages: str = Form("en"),
    target_language: str = Form("en"),
    summarize: bool = Form(False),
    mode: str = Form("extract")
):
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Save to temporary file
        temp_file = os.path.join(UPLOAD_DIR, "temp_image.png")
        image.save(temp_file)
        
        # Process image
        extracted_text, results = extract_text_from_image(temp_file, languages.split(','))
        confidence = sum(result[2] for result in results) / len(results) if results else 1.0
        
        # Initialize response variables
        translated_text = None
        summary = None
        analysis = None

        # Process based on mode
        if mode == "translate":
            if target_language != "en" and extracted_text:
                translated_text = translate_text(extracted_text, target_language)
        
        elif mode == "summarize" or summarize:
            summary = summarize_text(extracted_text)
        
        elif mode == "analysis":
            analysis = {
                "Document Type": "Image",
                "Character Count": len(extracted_text),
                "Word Count": len(extracted_text.split()),
                "Line Count": len(extracted_text.splitlines()),
                "Language": detect_language(extracted_text),
                "Key Topics": extract_key_topics(extracted_text),
                "Sentiment": analyze_sentiment(extracted_text),
                "Readability": calculate_readability(extracted_text),
                "Structure": analyze_text_structure(extracted_text)
            }
        
        # Clean up
        os.remove(temp_file)
        
        return ProcessingResponse(
            status="success",
            text=extracted_text,
            translated_text=translated_text,
            summary=summary,
            confidence=confidence,
            analysis=analysis
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/supported-languages")
async def get_supported_languages():
    return {
        "ocr_languages": ["en", "ta", "hi", "fr", "es", "de", "zh", "ja", "ko"],  # Add more as needed
        "translation_languages": ["en", "ta", "hi", "fr", "es", "de", "zh", "ja", "ko"]  # Add more as needed
    }

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

def detect_language(text):
    """Detect the language of the text."""
    try:
        return detect(text)
    except:
        return "unknown"

def extract_key_topics(text, num_topics=5):
    """Extract key topics from text using simple word frequency analysis."""
    # Remove common words and punctuation
    stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", 
                     "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 
                     'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 
                     'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 
                     'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
                     'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 
                     'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 
                     'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
                     'under', 'again', 'further', 'then', 'once'])
    
    # Tokenize and clean text
    words = re.findall(r'\b\w+\b', text.lower())
    words = [word for word in words if word not in stop_words and len(word) > 3]
    
    # Count word frequencies
    word_freq = Counter(words)
    
    # Get top topics
    topics = word_freq.most_common(num_topics)
    return [{"word": word, "frequency": freq} for word, freq in topics]

def analyze_sentiment(text):
    """Analyze the sentiment of the text using TextBlob."""
    try:
        analysis = TextBlob(text)
        # Get polarity (-1 to 1) and convert to percentage
        sentiment_score = (analysis.sentiment.polarity + 1) / 2 * 100
        
        # Determine sentiment category
        if sentiment_score >= 60:
            category = "Positive"
        elif sentiment_score <= 40:
            category = "Negative"
        else:
            category = "Neutral"
        
        return {
            "category": category,
            "score": round(sentiment_score, 2)
        }
    except:
        return {
            "category": "Unknown",
            "score": 50.0
        }

def calculate_readability(text):
    """Calculate readability metrics for the text."""
    try:
        # Count sentences, words, and syllables
        sentences = len(sent_tokenize(text))
        words = len(text.split())
        if words == 0:
            return {
                "flesch_score": 0,
                "grade_level": "N/A",
                "avg_words_per_sentence": 0
            }
        
        # Calculate Flesch Reading Ease score
        syllables = sum([len(re.findall(r'[aeiou]+', word.lower())) for word in text.split()])
        if syllables == 0:
            flesch_score = 0
        else:
            flesch_score = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
            flesch_score = max(0, min(100, flesch_score))  # Clamp between 0 and 100
        
        # Determine grade level
        if flesch_score >= 90:
            grade_level = "5th Grade"
        elif flesch_score >= 80:
            grade_level = "6th Grade"
        elif flesch_score >= 70:
            grade_level = "7th Grade"
        elif flesch_score >= 60:
            grade_level = "8th-9th Grade"
        elif flesch_score >= 50:
            grade_level = "10th-12th Grade"
        elif flesch_score >= 30:
            grade_level = "College"
        else:
            grade_level = "College Graduate"
        
        return {
            "flesch_score": round(flesch_score, 2),
            "grade_level": grade_level,
            "avg_words_per_sentence": round(words / sentences, 2)
        }
    except Exception as e:
        logger.error(f"Error calculating readability: {e}")
        return {
            "flesch_score": 0,
            "grade_level": "Unknown",
            "avg_words_per_sentence": 0
        }

def analyze_text_structure(text):
    """Analyze the structure of the text."""
    try:
        paragraphs = text.split('\n\n')
        sentences = sent_tokenize(text)
        words = text.split()
        lines = text.split('\n')
        
        # Basic statistics
        total_paragraphs = len(paragraphs)
        total_sentences = len(sentences)
        total_words = len(words)
        
        # Calculate averages
        avg_paragraph_length = total_words / total_paragraphs if total_paragraphs else 0
        avg_sentence_length = total_words / total_sentences if total_sentences else 0
        
        # Analyze paragraph distribution
        paragraph_lengths = [len(p.split()) for p in paragraphs]
        max_paragraph_length = max(paragraph_lengths) if paragraph_lengths else 0
        min_paragraph_length = min(paragraph_lengths) if paragraph_lengths else 0
        
        # Identify potential headings and sections
        potential_headings = []
        section_markers = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line:
                # Check for heading patterns
                is_heading = (
                    len(line.split()) <= 10 and  # Short line
                    not line[-1] in '.!?:,;' and  # No ending punctuation
                    (line.isupper() or  # ALL CAPS
                     line[0].isupper() or  # Starts with capital
                     any(c in line for c in ['#', '-', '=']))  # Common heading markers
                )
                
                if is_heading:
                    potential_headings.append(line)
                    section_markers.append(i)
        
        # Analyze document organization
        has_clear_sections = len(section_markers) > 0
        avg_section_length = (len(lines) / (len(section_markers) + 1)) if section_markers else len(lines)
        
        # Analyze paragraph coherence
        coherence_score = 0
        if total_paragraphs > 1:
            # Check if paragraphs are relatively consistent in length
            avg_length_diff = sum(abs(l - avg_paragraph_length) for l in paragraph_lengths) / total_paragraphs
            length_consistency = 1 - (avg_length_diff / avg_paragraph_length if avg_paragraph_length else 0)
            coherence_score = min(100, max(0, length_consistency * 100))
        
        return {
            "paragraphs": total_paragraphs,
            "avg_paragraph_length": round(avg_paragraph_length, 2),
            "max_paragraph_length": max_paragraph_length,
            "min_paragraph_length": min_paragraph_length,
            "avg_sentence_length": round(avg_sentence_length, 2),
            "potential_sections": len(potential_headings),
            "section_titles": potential_headings[:5],  # Show first 5 section titles
            "document_structure": {
                "has_clear_sections": has_clear_sections,
                "avg_section_length": round(avg_section_length, 2),
                "structure_score": round(coherence_score, 2),
                "organization_type": (
                    "Well Structured" if coherence_score >= 80
                    else "Moderately Structured" if coherence_score >= 60
                    else "Loosely Structured" if coherence_score >= 40
                    else "Poorly Structured"
                )
            }
        }
    except Exception as e:
        logger.error(f"Error analyzing text structure: {e}")
        return {
            "paragraphs": 0,
            "avg_paragraph_length": 0,
            "max_paragraph_length": 0,
            "min_paragraph_length": 0,
            "avg_sentence_length": 0,
            "potential_sections": 0,
            "section_titles": [],
            "document_structure": {
                "has_clear_sections": False,
                "avg_section_length": 0,
                "structure_score": 0,
                "organization_type": "Unknown"
            }
        }

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
