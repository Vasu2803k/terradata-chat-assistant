import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import asyncio
import pdfplumber

from scripts.log_config import get_logger


logger = get_logger(__name__)

# Set project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

class TextExtraction:
    def __init__(self, pdf_root_dir, output_dir):
        logger.info("---Entering TextExtraction.__init__---")
        self.pdf_root_dir = pdf_root_dir
        self.output_dir = output_dir
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("---End of TextExtraction.__init__---")

    def extract_pdf_metadata(self, pdf_path):
        logger.info("---Entering extract_pdf_metadata---")
        metadata = {}
        try:
            with pdfplumber.open(pdf_path) as pdf:
                meta = pdf.metadata or {}
                # Common fields: Title, Author, Subject, Producer, CreationDate, ModDate, etc.
                for key in meta:
                    metadata[key.lower()] = meta[key]
                logger.info(f"Extracted metadata for {pdf_path}: {metadata}")
        except Exception as e:
            logger.error(f"Error extracting metadata from {pdf_path}: {e}")
        logger.info("---End of extract_pdf_metadata---")
        return metadata

    async def process_pdf(self, pdf_path, page_number, loop):
        logger.info("---Entering process_pdf---")
        text = ""
        def _extract():
            local_text = ""
            with pdfplumber.open(pdf_path) as pdf:
                if page_number <= len(pdf.pages):
                    page = pdf.pages[page_number - 1]
                    local_text = page.extract_text() or ""
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            local_text += f"\nTABLE: {table}\n"
                else:
                    print("page_number exceeded")
            return local_text
        result = await loop.run_in_executor(None, _extract)
        logger.info("---End of process_pdf---")
        return result

    async def extract_text_and_tables(self, pdf_path, loop):
        logger.info("---Entering extract_text_and_tables---")
        if not pdf_path.exists():
            print(f"No pdf: {pdf_path}")
            return
        try:
            def _get_num_pages():
                with pdfplumber.open(pdf_path) as pdf:
                    return len(pdf.pages)
            pages = await loop.run_in_executor(None, _get_num_pages)
        except Exception as e:
            print(f"Pdf file cannot be opened: {e}")
            return
        # Extract metadata
        metadata = self.extract_pdf_metadata(pdf_path)
        extracted_text = ""
        for page_num in range(1, pages + 1):
            page_text = await self.process_pdf(pdf_path, page_num, loop)
            extracted_text += f"\n{page_text}\n"
        output_file = self.output_dir / f"{pdf_path.stem}.txt"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        def _write_file():
            logger.info("---Entering _write_file---")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(extracted_text)
            logger.info("---End of _write_file---")
        await loop.run_in_executor(None, _write_file)
        print(f"PDF '{pdf_path}' converted and saved as '{output_file}'")
        logger.info(f"Metadata for {pdf_path}: {metadata}")
        logger.info("---End of extract_text_and_tables---")
        return metadata

    async def process_pdf_files_async(self):
        logger.info("---Entering process_pdf_files_async---")
        pdf_files = []
        for root, dirs, files in os.walk(self.pdf_root_dir):
            if files:
                pdf_files.extend([Path(root) / file for file in files if file.endswith(".pdf")])
        loop = asyncio.get_running_loop()
        tasks = [self.extract_text_and_tables(pdf_path, loop) for pdf_path in pdf_files]
        metadatas = await asyncio.gather(*tasks)
        logger.info(f"Extracted metadata for all PDFs: {metadatas}")
        logger.info("---End of process_pdf_files_async---")

def text_extraction_tool(*args, **kwargs):
    logger.info("---Entering text_extraction_tool---")
    pdf_root_dir = PROJECT_ROOT / 'input'
    output_dir = PROJECT_ROOT / 'output'
    extractor = TextExtraction(pdf_root_dir, output_dir)
    asyncio.run(extractor.process_pdf_files_async())
    logger.info("---End of text_extraction_tool---")

# Example usage:
if __name__ == "__main__":
    text_extraction_tool()
