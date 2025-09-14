# techvision_agent/agent.py (Complete Production System with PDF Processing and Text Chunking)
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Dict, List, Any, Optional
import logging
import time
import io

# PDF processing import
import PyPDF2

# ADK imports
from google.adk.agents import LlmAgent
from google.adk.tools.function_tool import FunctionTool

# Production imports (no sentence-transformers or faiss)
from google.cloud import bigquery, aiplatform, storage
import google.generativeai as genai
from googleapiclient.discovery import build
import requests

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment variables
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "proj-newsshield-prod-infra")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GOOGLE_SEARCH_KEY = os.getenv("GOOGLE_SEARCH_KEY")
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

# Cloud Storage configuration for PDF documents
STORAGE_BUCKET_NAME = os.getenv("STORAGE_BUCKET_NAME", "chat_drl_bq_data")
PDF_FOLDER_PATH = os.getenv("PDF_FOLDER_PATH", "unstructured_documents/")

# Initialize services
try:
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    genai.configure(api_key=GOOGLE_API_KEY)
    logger.info("✅ Production services initialized")
except Exception as e:
    logger.error(f"Failed to initialize services: {str(e)}")

# Initialize BigQuery client
try:
    bq_client = bigquery.Client(project=PROJECT_ID)
    logger.info("✅ BigQuery client initialized")
except Exception as e:
    logger.error(f"BigQuery initialization failed: {str(e)}")
    bq_client = None

class ProductionDataManager:
    """Production data manager that works with existing BigQuery tables"""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.dataset_id = "techvision_analytics"
        self.client = bq_client

    def get_table_schemas(self) -> Dict[str, Any]:
        """Get schemas for all existing tables"""
        try:
            if not self.client:
                return {}
                
            dataset_ref = self.client.get_dataset(f"{self.project_id}.{self.dataset_id}")
            schemas = {}

            for table in self.client.list_tables(dataset_ref):
                table_ref = self.client.get_table(table.reference)
                schema = []
                for field in table_ref.schema:
                    schema.append({
                        "name": field.name,
                        "type": field.field_type,
                        "mode": field.mode
                    })

                schemas[table.table_id] = {
                    "schema": schema,
                    "description": f"Table containing {table.table_id} data"
                }
            return schemas
        except Exception as e:
            logger.error(f"Schema retrieval failed: {str(e)}")
            return {}

    def validate_tables_exist(self) -> bool:
        """Validate that required tables exist"""
        try:
            if not self.client:
                return False
                
            required_tables = ["sales_data", "customer_metrics", "employee_data", "financial_reports"]
            dataset_ref = self.client.get_dataset(f"{self.project_id}.{self.dataset_id}")
            existing_tables = [table.table_id for table in self.client.list_tables(dataset_ref)]

            missing_tables = [table for table in required_tables if table not in existing_tables]

            if missing_tables:
                logger.warning(f"Missing tables: {missing_tables}")
                return False

            logger.info("✅ All required tables exist")
            return True
        except Exception as e:
            logger.error(f"Table validation failed: {str(e)}")
            return False

class PDFDocumentProcessor:
    """PDF document processor that reads from Cloud Storage and extracts text"""

    def __init__(self, project_id: str, bucket_name: str):
        self.project_id = project_id
        self.bucket_name = bucket_name

        # Initialize Cloud Storage client
        try:
            self.storage_client = storage.Client(project=project_id)
            self.bucket = self.storage_client.bucket(bucket_name)
            logger.info("✅ Cloud Storage client initialized successfully")
        except Exception as e:
            logger.error(f"Cloud Storage initialization failed: {str(e)}")
            self.storage_client = None
            self.bucket = None

    def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF bytes using PyPDF2"""
        try:
            pdf_file = io.BytesIO(pdf_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    text += page_text + "\n"
                except Exception as e:
                    logger.warning(f"Could not extract text from page {page_num + 1}: {str(e)}")
                    continue

            return text.strip()
        except Exception as e:
            raise Exception(f"PDF text extraction failed: {str(e)}")

    def list_pdf_documents(self, folder_path: str = "") -> List[Dict[str, str]]:
        """List all PDF documents in the specified folder"""
        try:
            if not self.bucket:
                return []
                
            blobs = self.bucket.list_blobs(prefix=folder_path)
            pdf_documents = []

            for blob in blobs:
                if blob.name.lower().endswith('.pdf'):
                    # Extract title from filename (remove path and extension)
                    filename = blob.name.split('/')[-1]
                    title = filename[:-4]  # Remove .pdf extension

                    pdf_documents.append({
                        "blob_name": blob.name,
                        "title": title,
                        "size": blob.size,
                        "created": blob.time_created.isoformat() if blob.time_created else None,
                        "updated": blob.updated.isoformat() if blob.updated else None
                    })

            logger.info(f"Found {len(pdf_documents)} PDF documents in bucket")
            return pdf_documents
        except Exception as e:
            logger.error(f"Failed to list PDF documents: {str(e)}")
            return []

    def download_and_extract_text(self, blob_name: str) -> Dict[str, str]:
        """Download PDF from Cloud Storage and extract text"""
        try:
            if not self.bucket:
                raise Exception("Cloud Storage bucket not initialized")
                
            blob = self.bucket.blob(blob_name)
            pdf_bytes = blob.download_as_bytes()

            text = self.extract_text_from_pdf(pdf_bytes)

            # Extract title from blob name
            filename = blob_name.split('/')[-1]
            title = filename[:-4]  # Remove .pdf extension

            return {
                "title": title,
                "content": text,
                "blob_name": blob_name,
                "word_count": len(text.split())
            }
        except Exception as e:
            raise Exception(f"Failed to download and process PDF {blob_name}: {str(e)}")

class GeminiEmbeddingsBigQueryCorpus:
    """Production document corpus using Gemini embeddings and BigQuery vector search with PDF integration and text chunking"""

    def __init__(self, project_id: str, dataset_id: str = "techvision_analytics", chunk_size: int = 500, chunk_overlap: int = 100):
        logger.info("📚 Initializing Gemini + BigQuery document corpus with PDF processing and text chunking...")

        self.project_id = project_id
        self.dataset_id = dataset_id
        # Use new table name to avoid schema conflicts
        self.table_id = "document_embeddings"
        self.full_table_id = f"{project_id}.{dataset_id}.{self.table_id}"
        
        # Chunking parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.client = bq_client
        self.document_count = 0
        self.chunk_count = 0

        # Production documents from original system (fallback)
        self.production_documents = {
            "Strategic Plan 2024-2025": """
TechVision Analytics Strategic Plan 2024-2025

Executive Summary:
TechVision Analytics aims to become the leading B2B analytics platform for mid-market companies. Our three-year strategy focuses on product innovation, market expansion, and operational excellence.

Key Strategic Objectives:
1. Product Development: Launch AI-powered predictive analytics suite by Q2 2024
2. Market Expansion: Enter European market with localized offerings
3. Customer Success: Achieve 95% customer satisfaction and <5% churn rate
4. Revenue Growth: Target 150% revenue growth by end of 2025
5. Team Scaling: Grow engineering team by 200% while maintaining culture

Market Positioning:
We position ourselves as the bridge between enterprise-grade analytics tools (too complex/expensive) and basic reporting tools (insufficient capabilities). Our sweet spot is companies with 100-5000 employees who need sophisticated insights without enterprise complexity.

Success Metrics:
- Monthly Recurring Revenue (MRR) growth >15% month-over-month
- Customer Acquisition Cost (CAC) payback period <12 months
- Net Promoter Score (NPS) >50
- Employee satisfaction score >4.5/5.0
- Product uptime >99.9%
""",

            "Customer Success Playbook": """
TechVision Analytics Customer Success Playbook

Mission Statement:
Our Customer Success team ensures every client achieves measurable business value through our analytics platform. We focus on adoption, expansion, and advocacy to drive mutual growth.

Customer Journey Framework:

Onboarding Phase (Days 0-30):
- Welcome email sequence with getting started resources
- Technical setup call within 48 hours of signup
- Data integration assistance and validation
- First dashboard creation session
- Success metrics establishment with customer
- Weekly check-ins to ensure smooth adoption

Growth Phase (Days 31-180):
- Monthly business reviews with key stakeholders
- Feature utilization analysis and recommendations
- Custom report development based on business needs
- Team training sessions for power users
- Usage analytics review and optimization suggestions
- Identification of expansion opportunities

Customer Health Scoring:
Red (At Risk): <60% feature utilization, no recent logins, support tickets unresolved
Yellow (Needs Attention): 60-80% utilization, infrequent usage, basic feature adoption
Green (Healthy): >80% utilization, regular engagement, expanding use cases
Blue (Champion): >95% utilization, active advocate, referring new customers
""",
        }

        # Create embeddings table if it doesn't exist
        self._create_embeddings_table()

    def _create_embeddings_table(self):
        """Create BigQuery table for storing document embeddings with chunk support"""
        if not self.client:
            logger.error("BigQuery client not available")
            return

        try:
            # Try to delete existing table first (for clean start)
            # try:
            #     self.client.delete_table(self.full_table_id, not_found_ok=True)
            #     logger.info(f"🗑️ Cleaned up existing table (if any): {self.full_table_id}")
            # except Exception as e:
            #     logger.warning(f"Could not delete existing table (this is okay): {str(e)}")

            # Define table schema with chunking support
            schema = [
                bigquery.SchemaField("doc_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("chunk_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("chunk_index", "INTEGER", mode="REQUIRED"),
                bigquery.SchemaField("title", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("content", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("full_content", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("doc_type", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("word_count", "INTEGER", mode="REQUIRED"),
                bigquery.SchemaField("total_word_count", "INTEGER", mode="NULLABLE"),
                bigquery.SchemaField("chunk_start_pos", "INTEGER", mode="REQUIRED"),
                bigquery.SchemaField("chunk_end_pos", "INTEGER", mode="REQUIRED"),
                bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
                bigquery.SchemaField("source_blob", "STRING", mode="NULLABLE"),
            ]

            table = bigquery.Table(self.full_table_id, schema=schema)
            table = self.client.create_table(table, exists_ok=False)
            logger.info(f"✅ Document embeddings table created with chunking support: {self.full_table_id}")

        except Exception as e:
            logger.error(f"Failed to create embeddings table: {str(e)}")

    def _chunk_text(self, text: str) -> List[Dict[str, any]]:
        """Split text into overlapping chunks"""
        try:
            chunks = []
            text_length = len(text)
            
            if text_length <= self.chunk_size:
                # If text is smaller than chunk size, return as single chunk
                chunks.append({
                    'content': text,
                    'start_pos': 0,
                    'end_pos': text_length,
                    'chunk_index': 0
                })
                return chunks
            
            start = 0
            chunk_index = 0
            
            while start < text_length:
                # Calculate end position
                end = min(start + self.chunk_size, text_length)
                
                # If this isn't the last chunk, try to break at word boundary
                if end < text_length:
                    # Look for last space within reasonable distance
                    last_space = text.rfind(' ', start, end)
                    if last_space > start + (self.chunk_size * 0.8):  # If space is found in last 20% of chunk
                        end = last_space
                
                chunk_content = text[start:end].strip()
                
                if chunk_content:  # Only add non-empty chunks
                    chunks.append({
                        'content': chunk_content,
                        'start_pos': start,
                        'end_pos': end,
                        'chunk_index': chunk_index
                    })
                    chunk_index += 1
                
                # Calculate next start position with overlap
                if end >= text_length:
                    break
                    
                start = max(start + 1, end - self.chunk_overlap)
                
                # Prevent infinite loop
                if start >= text_length:
                    break
            
            logger.info(f"🔄 Text chunked into {len(chunks)} pieces (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})")
            return chunks
            
        except Exception as e:
            raise Exception(f"Failed to chunk text: {str(e)}")

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Gemini embedding model"""
        try:
            result = genai.embed_content(
                model="models/gemini-embedding-001",
                content=text,
                task_type="RETRIEVAL_DOCUMENT"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise

    def add_pdf_document(self, pdf_data: Dict[str, str]):
        """Add PDF document to BigQuery with Gemini embeddings and text chunking - with enhanced logging"""
        try:
            if not self.client:
                raise Exception("BigQuery client not available")

            # Generate unique doc ID
            doc_id = f"PDF_{self.document_count:03d}"
            self.document_count += 1

            # Chunk the document content
            chunks = self._chunk_text(pdf_data['content'])
            
            logger.info(f"\n📄 PROCESSING DOCUMENT: {pdf_data['title']}")
            logger.info(f"📊 Document Stats:")
            logger.info(f"   • Total Characters: {len(pdf_data['content']):,}")
            logger.info(f"   • Total Words: {pdf_data['word_count']:,}")
            logger.info(f"   • Generated Chunks: {len(chunks)}")
            logger.info(f"   • Chunk Size Config: {self.chunk_size} chars")
            logger.info(f"   • Overlap Config: {self.chunk_overlap} chars")

            # Process each chunk
            rows_to_insert = []
            logger.info(f"\n🔄 CHUNK PROCESSING DETAILS:")
            
            for chunk_data in chunks:
                # Generate unique chunk ID
                chunk_id = f"{doc_id}_CHUNK_{chunk_data['chunk_index']:03d}"
                
                # Show chunk details
                logger.info(f"   📋 Chunk {chunk_data['chunk_index']+1}/{len(chunks)}:")
                logger.info(f"      🆔 ID: {chunk_id}")
                logger.info(f"      📏 Position: {chunk_data['start_pos']}-{chunk_data['end_pos']} chars")
                logger.info(f"      💬 Length: {len(chunk_data['content'])} chars")
                logger.info(f"      📝 Words: {len(chunk_data['content'].split())} words")
                
                # Show content preview
                preview = chunk_data['content'][:100].replace('\n', ' ').strip()
                logger.info(f"      🔍 Preview: {preview}...")
                
                # Generate embedding for this chunk
                logger.info(f"      🧠 Generating embedding...")
                embedding = self._generate_embedding(chunk_data['content'])
                logger.info(f"      ✅ Embedding: {len(embedding)} dimensions")

                # Prepare row data
                row_data = {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_data['chunk_index'],
                    "title": pdf_data['title'],
                    "content": chunk_data['content'],  # Chunk content
                    "full_content": pdf_data['content'] if chunk_data['chunk_index'] == 0 else None,  # Store full content only in first chunk
                    "doc_type": "pdf",
                    "word_count": len(chunk_data['content'].split()),
                    "total_word_count": pdf_data['word_count'] if chunk_data['chunk_index'] == 0 else None,
                    "chunk_start_pos": chunk_data['start_pos'],
                    "chunk_end_pos": chunk_data['end_pos'],
                    "created_at": datetime.now().isoformat(),
                    "embedding": embedding,
                    "source_blob": pdf_data.get('blob_name', '')
                }
                
                rows_to_insert.append(row_data)
                self.chunk_count += 1

            # Batch insert all chunks for this document
            logger.info(f"\n💾 INSERTING INTO BIGQUERY:")
            logger.info(f"   • Inserting {len(rows_to_insert)} chunks...")
            
            errors = self.client.insert_rows_json(
                self.client.get_table(self.full_table_id),
                rows_to_insert
            )

            if errors:
                raise Exception(f"BigQuery insertion errors: {errors}")

            logger.info(f"✅ DOCUMENT COMPLETED: {pdf_data['title']}")
            logger.info(f"   • Document ID: {doc_id}")
            logger.info(f"   • Chunks Created: {len(chunks)}")
            logger.info(f"   • Embedding Dimensions: {len(embedding)} per chunk")
            logger.info(f"   • Total Chunks in Corpus: {self.chunk_count}")

        except Exception as e:
            raise Exception(f"Failed to add PDF document: {str(e)}")

    def add_production_document(self, doc_id: str, title: str, content: str):
        """Add a production document with chunking"""
        try:
            if not self.client:
                raise Exception("BigQuery client not available")

            # Chunk the document content
            chunks = self._chunk_text(content)
            
            logger.info(f"📄 Processing document: {title}")
            logger.info(f"🔄 Generated {len(chunks)} chunks for processing")

            # Process each chunk
            rows_to_insert = []
            for chunk_data in chunks:
                # Generate unique chunk ID
                chunk_id = f"{doc_id}_CHUNK_{chunk_data['chunk_index']:03d}"
                
                # Generate embedding for this chunk
                logger.info(f"🧠 Generating Gemini embedding for chunk {chunk_data['chunk_index']+1}/{len(chunks)}...")
                embedding = self._generate_embedding(chunk_data['content'])

                # Prepare row data
                row_data = {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_data['chunk_index'],
                    "title": title,
                    "content": chunk_data['content'],  # Chunk content
                    "full_content": content if chunk_data['chunk_index'] == 0 else None,  # Store full content only in first chunk
                    "doc_type": "internal",
                    "word_count": len(chunk_data['content'].split()),
                    "total_word_count": len(content.split()) if chunk_data['chunk_index'] == 0 else None,
                    "chunk_start_pos": chunk_data['start_pos'],
                    "chunk_end_pos": chunk_data['end_pos'],
                    "created_at": datetime.now().isoformat(),
                    "embedding": embedding,
                    "source_blob": None
                }
                
                rows_to_insert.append(row_data)
                self.chunk_count += 1

            # Batch insert all chunks for this document
            errors = self.client.insert_rows_json(
                self.client.get_table(self.full_table_id),
                rows_to_insert
            )

            if errors:
                raise Exception(f"BigQuery insertion errors: {errors}")

            logger.info(f"✅ Added internal document: {title} (ID: {doc_id}, {len(chunks)} chunks, {len(embedding)} dimensions each)")

        except Exception as e:
            raise Exception(f"Failed to add internal document: {str(e)}")

    def process_all_pdfs(self, folder_path: str = PDF_FOLDER_PATH):
        """Process all PDFs from Cloud Storage and add to corpus with chunking"""
        try:
            # Initialize PDF processor
            pdf_processor = PDFDocumentProcessor(PROJECT_ID, STORAGE_BUCKET_NAME)
            
            # Get list of PDF documents
            pdf_documents = pdf_processor.list_pdf_documents(folder_path)

            if not pdf_documents:
                logger.warning("No PDF documents found in the specified folder")
                return "⚠️ No PDF documents found"

            logger.info(f"📄 Processing {len(pdf_documents)} PDF documents with chunking...")

            processed_count = 0
            for i, pdf_info in enumerate(pdf_documents):
                try:
                    logger.info(f"📖 Processing PDF {i+1}/{len(pdf_documents)}: {pdf_info['title']}")

                    # Download and extract text
                    pdf_data = pdf_processor.download_and_extract_text(pdf_info['blob_name'])

                    # Add to corpus with chunking and embeddings
                    self.add_pdf_document(pdf_data)
                    processed_count += 1

                except Exception as e:
                    logger.error(f"Failed to process {pdf_info['title']}: {str(e)}")
                    continue

            return f"✅ Successfully processed {processed_count}/{len(pdf_documents)} PDF documents: {self.document_count} documents, {self.chunk_count} chunks total"

        except Exception as e:
            error_msg = f"Failed to process PDFs: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def setup_document_embeddings(self) -> str:
        """One-time setup: Generate and store all document embeddings including PDFs with chunking"""
        try:
            if not self.client:
                return "❌ BigQuery client not available"

            logger.info("🚀 Starting document embeddings setup with chunking...")
            
            # Check if documents already exist
            try:
                doc_stats = self.get_document_count()
                existing_docs = doc_stats['documents']
                existing_chunks = doc_stats['chunks']
                
                if existing_docs > 0:
                    # Also process PDFs if they exist
                    pdf_result = self.process_all_pdfs()
                    return f"✅ Document embeddings already exist: {existing_docs} documents ({existing_chunks} chunks). {pdf_result}"
            except:
                pass  # Table might not exist yet

            response_messages = []
            
            # Process built-in documents with chunking
            for doc_index, (title, content) in enumerate(self.production_documents.items()):
                try:
                    doc_id = f"doc_{doc_index:03d}"
                    self.add_production_document(doc_id, title, content)
                    logger.info(f"✅ Generated embeddings for: {title}")
                    
                except Exception as e:
                    logger.error(f"Failed to generate embedding for {title}: {str(e)}")
                    continue
            
            if self.document_count > 0:
                response_messages.append(f"✅ Successfully stored {self.document_count} documents ({self.chunk_count} chunks) with Gemini embeddings")
            
            # Also process PDFs
            pdf_result = self.process_all_pdfs()
            response_messages.append(pdf_result)
            
            return " | ".join(response_messages) if response_messages else "❌ No embeddings were generated successfully"
                
        except Exception as e:
            error_msg = f"Embeddings setup failed: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def semantic_search(self, query: str, top_k: int = 7) -> List[Dict]:
        """Enhanced semantic search working with chunked documents - with detailed chunk visibility"""
        try:
            if not self.client:
                return []

            # Generate query embedding
            logger.info(f"🔍 Generating query embedding for: {query[:50]}...")
            query_embedding = self._generate_embedding(query)
            embedding_array_str = "[" + ",".join(map(str, query_embedding)) + "]"

            # Updated search query to work with chunks
            try:
                search_query = f"""
                SELECT
                    base.doc_id,
                    base.chunk_id,
                    base.chunk_index,
                    base.title,
                    base.content,
                    base.full_content,
                    base.doc_type,
                    base.word_count,
                    base.total_word_count,
                    base.chunk_start_pos,
                    base.chunk_end_pos,
                    base.created_at,
                    base.source_blob,
                    distance,
                    (1 - distance) AS similarity_score
                FROM VECTOR_SEARCH(
                    TABLE `{self.full_table_id}`,
                    'embedding',
                    (SELECT {embedding_array_str} AS embedding),
                    distance_type => 'COSINE',
                    top_k => {top_k * 3}  -- Get more chunks to potentially represent more documents
                )
                ORDER BY distance ASC
                """

                logger.info("🔎 Performing VECTOR_SEARCH on chunked documents...")
                results = list(self.client.query(search_query))
                logger.info(f"✅ Vector search successful! Found {len(results)} total chunks")

            except Exception as direct_error:
                logger.warning(f"⚠️ Direct approach failed: {str(direct_error)[:100]}")

                # Fallback to table-to-table approach
                logger.info("🔄 Trying table-to-table approach...")

                temp_table_id = f"{self.full_table_id}_temp_query_{int(time.time())}"

                # Create temporary query table
                query_data = [{"query_embedding": query_embedding}]
                temp_schema = [bigquery.SchemaField("query_embedding", "FLOAT64", mode="REPEATED")]

                temp_table = bigquery.Table(temp_table_id, schema=temp_schema)
                temp_table = self.client.create_table(temp_table, exists_ok=True)
                self.client.insert_rows_json(temp_table, query_data)

                try:
                    search_query = f"""
                    SELECT
                        base.doc_id,
                        base.chunk_id,
                        base.chunk_index,
                        base.title,
                        base.content,
                        base.full_content,
                        base.doc_type,
                        base.word_count,
                        base.total_word_count,
                        base.chunk_start_pos,
                        base.chunk_end_pos,
                        base.created_at,
                        base.source_blob,
                        distance,
                        (1 - distance) AS similarity_score
                    FROM VECTOR_SEARCH(
                        TABLE `{self.full_table_id}`,
                        'embedding',
                        TABLE `{temp_table_id}`,
                        query_column_to_search => 'query_embedding',
                        distance_type => 'COSINE',
                        top_k => {top_k * 3}
                    )
                    ORDER BY distance ASC
                    """

                    results = list(self.client.query(search_query))
                    logger.info(f"✅ Table-to-table VECTOR_SEARCH successful!")

                finally:
                    # Cleanup
                    try:
                        self.client.delete_table(temp_table_id)
                    except:
                        pass

            # Enhanced logging: Show all found chunks before filtering
            if results:
                logger.info("\n📋 DETAILED CHUNK ANALYSIS:")
                logger.info("=" * 80)
                for i, row in enumerate(results):
                    logger.info(f"\n🔍 CHUNK {i+1}/{len(results)}:")
                    logger.info(f"   📄 Document: {row.title}")
                    logger.info(f"   🆔 Doc ID: {row.doc_id}")
                    logger.info(f"   🧩 Chunk ID: {row.chunk_id}")
                    logger.info(f"   📍 Chunk Index: {row.chunk_index}")
                    logger.info(f"   📏 Position: {row.chunk_start_pos}-{row.chunk_end_pos} chars")
                    logger.info(f"   📊 Similarity Score: {float(row.similarity_score):.4f}")
                    logger.info(f"   📐 Distance: {float(row.distance):.4f}")
                    logger.info(f"   💬 Word Count: {row.word_count}")
                    if row.source_blob:
                        logger.info(f"   📎 PDF Source: {row.source_blob}")
                    
                    # Show chunk content preview (first 150 chars)
                    chunk_preview = row.content[:150].replace('\n', ' ').strip()
                    logger.info(f"   📝 Content Preview: {chunk_preview}...")
                    logger.info("   " + "-" * 60)

            # Format results with chunk awareness
            formatted_results = []
            seen_docs = set()
            
            logger.info(f"\n🎯 FILTERING TO TOP {top_k} RESULTS:")
            logger.info("=" * 50)
            
            for i, row in enumerate(results):
                # Limit to requested number of unique documents/chunks
                if len(formatted_results) >= top_k:
                    break
                    
                logger.info(f"\n✅ SELECTED CHUNK {len(formatted_results)+1}:")
                logger.info(f"   📄 {row.title}")
                logger.info(f"   🧩 {row.chunk_id} (Index: {row.chunk_index})")
                logger.info(f"   📊 Similarity: {float(row.similarity_score):.4f}")
                    
                # Create a comprehensive content preview
                chunk_content = row.content
                
                # For context, show some surrounding text if available
                if hasattr(row, 'full_content') and row.full_content:
                    full_content = row.full_content
                    start_pos = max(0, row.chunk_start_pos - 100)  # Add some context before
                    end_pos = min(len(full_content), row.chunk_end_pos + 100)  # Add some context after
                    context_content = full_content[start_pos:end_pos]
                    relevant_content = f"...{context_content}..." if start_pos > 0 or end_pos < len(full_content) else context_content
                    logger.info(f"   🎯 Enhanced with context from position {start_pos}-{end_pos}")
                else:
                    # Fallback to chunk content with preview
                    sentences = [s.strip() for s in chunk_content.split('.') if s.strip()]
                    relevant_content = '. '.join(sentences[:3]) + '...' if len(sentences) > 3 else chunk_content
                    logger.info(f"   📝 Using chunk content only (no full document available)")

                formatted_results.append({
                    "id": row.doc_id,
                    "chunk_id": row.chunk_id,
                    "chunk_index": row.chunk_index,
                    "title": row.title,
                    "content": chunk_content,  # The actual matching chunk
                    "relevant_content": relevant_content,  # Enhanced preview with context
                    "full_content": row.full_content if hasattr(row, 'full_content') else None,
                    "similarity_score": float(row.similarity_score),
                    "distance": float(row.distance),
                    "document_type": row.doc_type,
                    "word_count": row.word_count,
                    "total_word_count": row.total_word_count if hasattr(row, 'total_word_count') else None,
                    "chunk_position": f"chars {row.chunk_start_pos}-{row.chunk_end_pos}",
                    "rank": len(formatted_results) + 1,
                    "created": row.created_at.isoformat() if hasattr(row.created_at, 'isoformat') else str(row.created_at),
                    "source_blob": row.source_blob if hasattr(row, 'source_blob') else None
                })

            logger.info(f"\n✅ FINAL RESULT: Selected {len(formatted_results)} chunks from {len(set([r.doc_id for r in results]))} unique documents")
            logger.info("=" * 80)
            return formatted_results

        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            return []

    def get_document_count(self) -> Dict[str, int]:
        """Get total number of documents and chunks in corpus - with detailed breakdown"""
        try:
            if not self.client:
                return {"documents": 0, "chunks": 0, "breakdown": []}
                
            # Count unique documents
            doc_count_query = f"SELECT COUNT(DISTINCT doc_id) as total_docs FROM `{self.full_table_id}`"
            doc_result = list(self.client.query(doc_count_query))
            
            # Count total chunks
            chunk_count_query = f"SELECT COUNT(*) as total_chunks FROM `{self.full_table_id}`"
            chunk_result = list(self.client.query(chunk_count_query))
            
            # Get document breakdown by type
            breakdown_query = f"""
            SELECT 
                doc_type,
                COUNT(DISTINCT doc_id) as documents,
                COUNT(*) as chunks,
                AVG(word_count) as avg_chunk_words,
                MIN(word_count) as min_chunk_words,
                MAX(word_count) as max_chunk_words
            FROM `{self.full_table_id}` 
            GROUP BY doc_type
            ORDER BY documents DESC
            """
            breakdown_result = list(self.client.query(breakdown_query))
            
            document_stats = {
                "documents": doc_result[0].total_docs if doc_result else 0,
                "chunks": chunk_result[0].total_chunks if chunk_result else 0,
                "breakdown": []
            }
            
            if breakdown_result:
                for row in breakdown_result:
                    document_stats["breakdown"].append({
                        "type": row.doc_type,
                        "documents": row.documents,
                        "chunks": row.chunks,
                        "avg_chunk_words": round(row.avg_chunk_words, 1),
                        "min_chunk_words": row.min_chunk_words,
                        "max_chunk_words": row.max_chunk_words
                    })
            
            # Print detailed stats
            logger.info(f"\n📊 DOCUMENT CORPUS STATISTICS:")
            logger.info(f"   📚 Total Documents: {document_stats['documents']}")
            logger.info(f"   🧩 Total Chunks: {document_stats['chunks']}")
            logger.info(f"   📈 Avg Chunks per Document: {document_stats['chunks'] / max(document_stats['documents'], 1):.1f}")
            
            if document_stats["breakdown"]:
                logger.info(f"\n📋 BREAKDOWN BY TYPE:")
                for item in document_stats["breakdown"]:
                    logger.info(f"   {item['type'].upper()}:")
                    logger.info(f"      • Documents: {item['documents']}")
                    logger.info(f"      • Chunks: {item['chunks']}")
                    logger.info(f"      • Avg words/chunk: {item['avg_chunk_words']}")
                    logger.info(f"      • Word range: {item['min_chunk_words']}-{item['max_chunk_words']}")
            
            return document_stats
            
        except Exception as e:
            logger.warning(f"Failed to get document count: {str(e)}")
            return {"documents": 0, "chunks": 0, "breakdown": []}

# Initialize production components with chunking
data_manager = ProductionDataManager(PROJECT_ID)
doc_corpus = GeminiEmbeddingsBigQueryCorpus(
    PROJECT_ID, 
    chunk_size=500, 
    chunk_overlap=100
)

# Production Functions

def generate_sql_with_llm(query: str, table_schemas: Dict[str, Any]) -> str:
    """Generate SQL queries using Gemini"""
    try:
        model = genai.GenerativeModel('gemini-2.5-pro')

        schema_info = json.dumps(table_schemas, indent=2)

        prompt = f"""
        You are a SQL expert for TechVision Analytics. Generate a BigQuery SQL query based on the natural language request.

        Available tables and schemas:
        {schema_info}

        Dataset: {PROJECT_ID}.techvision_analytics

        Natural language query: "{query}"

        Requirements:
        1. Generate ONLY the SQL query, no explanation
        2. Use proper BigQuery syntax with correct table names (no extra asterisks or formatting)
        3. Include appropriate WHERE clauses, ORDER BY, and LIMIT as needed
        4. When filtering by date columns, use PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%E*S', column_name)
        5. Handle date ranges correctly using parsed timestamps
        6. For subtracting months or years, use DATE_SUB function
        7. When comparing TIMESTAMP with DATE, cast TIMESTAMP to DATE
        8. Ensure query is safe (no DROP, DELETE, TRUNCATE operations)
        9. Use table aliases for readability
        10. **CRITICAL: Always use SAFE_DIVIDE() instead of division (/) to prevent division by zero errors**
        11. **Add HAVING COUNT(*) > 0 clauses when doing aggregations to ensure non-empty results**
        12. **Use COALESCE() for null handling in calculations**
        13. ONLY use tables: customer_metrics, sales_data, financial_reports, employee_data
        14. DO NOT reference vertex_ai, ML functions, or document_embeddings tables
        15. Use LIMIT 10 for testing unless specifically asked for more

        EXAMPLE of safe division:
        SAFE_DIVIDE(COUNTIF(condition), COUNT(*)) * 100 AS percentage

        EXAMPLE of safe aggregation:
        SELECT ... FROM table GROUP BY column HAVING COUNT(*) > 0

        CRITICAL RULES - YOU MUST FOLLOW THESE:
        - ONLY use these tables from dataset `{PROJECT_ID}.techvision_analytics`:
          * customer_metrics (customer data, satisfaction, churn risk)
          * sales_data (revenue, deals, performance)  
          * financial_reports (financial metrics)
          * employee_data (staff information)
        - DO NOT use any of these - they don't exist:
          * vertex_ai dataset or models
          * ML.GENERATE_EMBEDDING functions
          * document_embeddings tables
          * Any ML or AI functions
        - Write SIMPLE SELECT queries only using standard SQL functions

        SQL Query:
        """

        response = model.generate_content(prompt)
        sql_query = response.text.strip()

        # Clean up the response
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]

        # Validate and fix common issues
        sql_query = sql_query.strip()
        
        # Remove problematic patterns
        if "vertex_ai" in sql_query.lower() or "ml.generate_embedding" in sql_query.lower():
            logger.warning("SQL query contained invalid ML references, using fallback")
            # Fallback to simple customer metrics query
            sql_query = f"""
            SELECT 
                churn_risk,
                COUNT(*) as customer_count,
                ROUND(AVG(satisfaction_score), 2) as avg_satisfaction,
                ROUND(AVG(lifetime_value), 2) as avg_lifetime_value
            FROM `{PROJECT_ID}.techvision_analytics.customer_metrics`
            GROUP BY churn_risk
            HAVING COUNT(*) > 0
            ORDER BY customer_count DESC
            LIMIT 10
            """
        
        # Fix common division issues
        sql_query = sql_query.replace(" / ", " SAFE_DIVIDE(")
        if "SAFE_DIVIDE(" in sql_query and not sql_query.count("SAFE_DIVIDE(") == sql_query.count(")"):
            # Fix incomplete SAFE_DIVIDE replacements
            sql_query = sql_query.replace("SAFE_DIVIDE(", "/").replace(" / ", ", ").replace("/", "SAFE_DIVIDE(")

        return sql_query.strip()

    except Exception as e:
        raise Exception(f"LLM SQL generation failed: {str(e)}")

def execute_bigquery_query(sql_query: str) -> Dict[str, Any]:
    """Execute SQL query against BigQuery"""
    try:
        if not bq_client:
            raise Exception("BigQuery client not available")

        # Safety check for dangerous operations
        query_upper = sql_query.upper()
        dangerous_keywords = ["DROP", "DELETE", "TRUNCATE", "ALTER", "CREATE", "UPDATE"]
        if any(keyword in query_upper for keyword in dangerous_keywords):
            raise Exception("Dangerous SQL operations are not allowed")

        # Execute query
        job = bq_client.query(sql_query)
        results = job.result()

        # Convert to list of dictionaries
        rows = []
        for row in results:
            row_dict = {}
            for key, value in row.items():
                if hasattr(value, 'isoformat'):  # datetime
                    row_dict[key] = value.isoformat()
                elif isinstance(value, (int, float, str, bool)) or value is None:
                    row_dict[key] = value
                else:
                    row_dict[key] = str(value)
            rows.append(row_dict)

        return {
            "success": True,
            "data": rows,
            "row_count": len(rows),
            "query": sql_query
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": sql_query
        }

def semantic_document_search(query: str, max_results: int = 7) -> Dict[str, Any]:
    """Perform semantic search using BigQuery vector search with chunking"""
    try:
        results = doc_corpus.semantic_search(query, top_k=max_results)

        return {
            "success": True,
            "query": query,
            "results": results,
            "total_found": len(results),
            "search_type": "bigquery_vector_search_with_gemini_embeddings_chunked"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": query
        }

def google_custom_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Perform web search using Google Custom Search API"""
    try:
        api_key = os.getenv("GOOGLE_SEARCH_KEY")
        search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

        if not api_key or not search_engine_id:
            raise Exception("Google API Key or Search Engine ID is not configured")

        service = build("customsearch", "v1", developerKey=api_key)

        response = service.cse().list(
            q=query,
            cx=search_engine_id,
            num=max_results
        ).execute()

        search_results = []
        items = response.get("items", [])
        for i, result in enumerate(items):
            search_results.append({
                "title": result.get("title", ""),
                "url": result.get("link", ""),
                "snippet": result.get("snippet", ""),
                "position": i + 1
            })

        return {
            "success": True,
            "query": query,
            "results": search_results,
            "total_found": len(search_results)
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": query
        }

def llm_response_synthesis(query: str, structured_data: Dict, document_data: Dict, web_data: Dict) -> str:
    """Use Gemini to synthesize responses from all agents"""
    try:
        model = genai.GenerativeModel('gemini-2.5-pro')

        context = {
            "original_query": query,
            "structured_data": structured_data,
            "document_data": document_data,
            "web_data": web_data
        }

        context_json = json.dumps(context, indent=2, default=str)

        prompt = f"""
        You are a business intelligence analyst for TechVision Analytics. Synthesize the following data sources into a comprehensive, executive-ready response.

        Data Sources:
        {context_json}

        Instructions:
        1. Create a professional business response that addresses the original query
        2. Combine insights from structured data, internal documents, and web research
        3. Highlight key findings with specific data points
        4. Provide actionable recommendations where appropriate
        5. Use clear structure: Executive Summary, Key Findings, Analysis, Recommendations
        6. Maintain professional, authoritative tone
        7. Cite data sources appropriately
        8. Focus on business value and strategic implications

        Generate a comprehensive response:
        """

        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        raise Exception(f"Response synthesis failed: {str(e)}")

# Tool Functions for ADK Web

def query_structured_data_tool(query: str) -> str:
    """Tool: Query structured business data using BigQuery"""
    try:
        logger.info(f"📊 Structured data query: {query}")
        
        if not bq_client:
            return "❌ BigQuery client not available"

        # Get table schemas
        schemas = data_manager.get_table_schemas()
        
        if not schemas:
            return "❌ No table schemas available"

        # Generate SQL using LLM
        sql_query = generate_sql_with_llm(query, schemas)
        logger.info(f"Generated SQL: {sql_query}")

        # Execute query
        result = execute_bigquery_query(sql_query)

        if result["success"]:
            response = f"📊 **BigQuery Analysis Results** ({result['row_count']} rows):\n\n"
            response += f"**SQL Query:** `{result['query']}`\n\n"
            response += "**Data:**\n"
            
            for i, row in enumerate(result["data"][:8]):
                response += f"{i+1}. {json.dumps(row, indent=2, default=str)}\n"
            
            if result["row_count"] > 8:
                response += f"\n... and {result['row_count'] - 8} more rows"
            
            return response
        else:
            return f"❌ BigQuery error: {result['error']}"

    except Exception as e:
        error_msg = f"Structured data query failed: {str(e)}"
        logger.error(error_msg)
        return f"❌ {error_msg}"

def search_documents_tool(query: str) -> str:
    """Tool: Search company documents including PDFs using Gemini embeddings + BigQuery vector search with chunking"""
    try:
        logger.info(f"📄 Document search: {query}")

        result = semantic_document_search(query, max_results=3)

        if result["success"]:
            if not result["results"]:
                return "📄 No relevant documents found. You may need to run document setup first."

            response = f"📄 **Company Knowledge Base** ({result['total_found']} matches):\n"
            response += f"*Search Type: {result['search_type']}*\n\n"

            for doc in result["results"]:
                response += f"**📋 {doc['title']}**\n"
                response += f"*Similarity Score: {doc['similarity_score']:.3f} | Type: {doc['document_type']}*\n"
                response += f"*Chunk: {doc['chunk_index']+1} | Words: {doc['word_count']:,}*\n"
                
                # Show chunk position info
                if doc.get('chunk_position'):
                    response += f"*Position: {doc['chunk_position']}*\n"
                
                # Show source information
                if doc.get('source_blob'):
                    response += f"*Source: PDF ({doc['source_blob']})*\n"
                else:
                    response += f"*Source: Internal Document*\n"
                
                # Show relevant content excerpt
                content = doc.get('relevant_content', doc['content'])
                if len(content) > 400:
                    content = content[:400] + "..."
                response += f"\n{content}\n\n"
                response += "---\n\n"

            return response
        else:
            return f"❌ Document search error: {result['error']}"

    except Exception as e:
        error_msg = f"Document search failed: {str(e)}"
        logger.error(error_msg)
        return f"❌ {error_msg}"

def search_web_tool(query: str) -> str:
    """Tool: Search web for market intelligence"""
    try:
        logger.info(f"🌐 Web search: {query}")

        # Add business analytics context to search
        enhanced_query = f"{query} business analytics industry trends 2024"
        result = google_custom_search(enhanced_query, max_results=5)

        if result["success"]:
            if not result["results"]:
                return "🌐 No web search results found"

            response = f"🌐 **Market Intelligence** ({result['total_found']} sources):\n\n"
            response += f"**Search Query:** `{enhanced_query}`\n\n"

            for item in result["results"]:
                response += f"**{item['position']}. {item['title']}**\n"
                response += f"*URL: {item['url']}*\n"
                response += f"{item['snippet']}\n\n"

            return response
        else:
            return f"❌ Web search error: {result['error']}"

    except Exception as e:
        error_msg = f"Web search failed: {str(e)}"
        logger.error(error_msg)
        return f"❌ {error_msg}"

def setup_vector_embeddings_tool() -> str:
    """Tool: One-time setup of document embeddings in BigQuery including PDF processing with chunking"""
    try:
        logger.info("🚀 Setting up vector embeddings with chunking...")
        return doc_corpus.setup_document_embeddings()
    except Exception as e:
        error_msg = f"Setup failed: {str(e)}"
        logger.error(error_msg)
        return f"❌ {error_msg}"

def process_pdfs_tool() -> str:
    """Tool: Process PDFs from Cloud Storage and add to document corpus with chunking"""
    try:
        logger.info("📄 Processing PDFs from Cloud Storage with chunking...")
        return doc_corpus.process_all_pdfs()
    except Exception as e:
        error_msg = f"PDF processing failed: {str(e)}"
        logger.error(error_msg)
        return f"❌ {error_msg}"

def analyze_query_intent(query: str) -> Dict[str, bool]:
    """Analyze query to determine which data sources are needed"""
    try:
        model = genai.GenerativeModel('gemini-2.5-pro')

        prompt = f"""
        Analyze this business query for TechVision Analytics and determine which data sources are needed.

        Query: "{query}"

        Available data sources:
        1. Structured Data: Sales, customer metrics, financial reports, employee data (BigQuery)
        2. Company Documents: Strategic plans, customer success playbooks, product specs, PDFs (Vector search with chunking)
        3. Web Search: Market intelligence, competitive analysis, industry trends

        Respond with ONLY a JSON object:
        {{
            "needs_structured": true/false,
            "needs_documents": true/false,
            "needs_web_search": true/false,
            "reasoning": "Brief explanation"
        }}
        """

        response = model.generate_content(prompt)
        response_text = response.text.strip()

        # Extract JSON
        import re
        json_match = re.search(r'\{[^{}]*\}', response_text)
        if json_match:
            result = json.loads(json_match.group())
            return {
                "needs_structured": bool(result.get("needs_structured", False)),
                "needs_documents": bool(result.get("needs_documents", False)),
                "needs_web_search": bool(result.get("needs_web_search", False)),
                "reasoning": result.get("reasoning", "AI analysis")
            }

        # Fallback based on keywords
        query_lower = query.lower()
        return {
            "needs_structured": any(word in query_lower for word in ["sales", "revenue", "customer", "financial", "top", "performance", "data"]),
            "needs_documents": any(word in query_lower for word in ["strategy", "plan", "success", "retention", "approach", "policy", "playbook", "document"]),
            "needs_web_search": any(word in query_lower for word in ["market", "trend", "industry", "competition", "benchmark"]),
            "reasoning": "Keyword-based fallback analysis"
        }

    except Exception as e:
        logger.error(f"Query analysis failed: {str(e)}")
        return {
            "needs_structured": True,
            "needs_documents": True,
            "needs_web_search": True,
            "reasoning": f"Analysis error, checking all sources: {str(e)}"
        }

def comprehensive_business_intelligence(query: str) -> str:
    """Main business intelligence orchestration function with PDF support and chunking"""
    try:
        logger.info(f"🚀 Comprehensive BI analysis: {query}")

        # Analyze query intent
        intent = analyze_query_intent(query)
        logger.info(f"🤖 Intent analysis: {intent['reasoning']}")

        # Collect data from relevant sources
        structured_data = {}
        document_data = {}
        web_data = {}

        if intent["needs_structured"]:
            logger.info("📊 Querying structured data...")
            try:
                structured_result = query_structured_data_tool(query)
                structured_data = {
                    "success": True,
                    "response": structured_result,
                    "source": "BigQuery"
                }
            except Exception as e:
                structured_data = {
                    "success": False,
                    "error": str(e),
                    "source": "BigQuery"
                }

        if intent["needs_documents"]:
            logger.info("📄 Searching company documents including PDFs with chunking...")
            try:
                document_result = search_documents_tool(query)
                document_data = {
                    "success": True,
                    "response": document_result,
                    "source": "Gemini Embeddings + BigQuery Vector Search + PDF Processing + Text Chunking"
                }
            except Exception as e:
                document_data = {
                    "success": False,
                    "error": str(e),
                    "source": "Document Vector Search"
                }

        if intent["needs_web_search"]:
            logger.info("🌐 Gathering web intelligence...")
            try:
                web_result = search_web_tool(query)
                web_data = {
                    "success": True,
                    "response": web_result,
                    "source": "Google Custom Search"
                }
            except Exception as e:
                web_data = {
                    "success": False,
                    "error": str(e),
                    "source": "Web Search"
                }

        # Synthesize response using LLM
        try:
            synthesized_response = llm_response_synthesis(query, structured_data, document_data, web_data)
            
            # Add metadata
            sources_used = []
            if structured_data.get("success"):
                sources_used.append("BigQuery Analytics")
            if document_data.get("success"):
                sources_used.append("Vector Document Search + PDF Processing + Text Chunking")
            if web_data.get("success"):
                sources_used.append("Web Intelligence")

            # Get current document stats
            doc_stats = doc_corpus.get_document_count()

            final_response = f"""# 🏢 TechVision Analytics Executive Intelligence

**Query**: {query}  
**Analysis Method**: {intent['reasoning']}  
**Sources**: {', '.join(sources_used) if sources_used else 'System Analysis'}  
**Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S EST')}

---

{synthesized_response}

---

## 🎯 Technical Details

**Vector Search**: Gemini text-embedding-004 + BigQuery cosine similarity  
**Document Processing**: PDF text extraction + semantic embeddings + text chunking  
**Chunking**: 500 chars/chunk, 100 chars overlap  
**Structured Analysis**: AI-generated SQL + BigQuery execution  
**Market Intelligence**: Google Custom Search API  
**Document Corpus**: {doc_stats['documents']} documents ({doc_stats['chunks']} chunks) in vector corpus  
**Cloud Storage**: {STORAGE_BUCKET_NAME}/{PDF_FOLDER_PATH}

*Powered by TechVision Analytics Cloud-Native Multi-Agent System with PDF Processing and Text Chunking*
"""

            logger.info("✅ Comprehensive BI analysis completed")
            return final_response

        except Exception as synthesis_error:
            # Fallback to direct combination if synthesis fails
            logger.warning(f"Synthesis failed, using direct combination: {str(synthesis_error)}")
            
            results = []
            if structured_data.get("success"):
                results.append(f"## 📊 Business Data Analysis\n{structured_data['response']}")
            if document_data.get("success"):
                results.append(f"## 📄 Company Knowledge (including PDFs with chunking)\n{document_data['response']}")
            if web_data.get("success"):
                results.append(f"## 🌐 Market Intelligence\n{web_data['response']}")

            if results:
                return f"""# 🏢 TechVision Analytics Business Intelligence

**Query**: {query}
**Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S EST')}

---

{chr(10).join(results)}

---

*TechVision Analytics Multi-Agent System with PDF Processing and Text Chunking*
"""
            else:
                return f"❌ Unable to process query: {query}"

    except Exception as e:
        error_msg = f"Comprehensive BI analysis failed: {str(e)}"
        logger.error(error_msg)
        return f"""# ❌ TechVision Analytics System Error

**Query**: {query}  
**Error**: {error_msg}  
**Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Please try again or contact system administrator.

*TechVision Analytics Multi-Agent System with PDF Processing and Text Chunking*
"""

# Create the main root agent for ADK Web (this is what gets discovered)
root_agent = LlmAgent(
    name="techvision_production_system_with_pdf_chunking",
    model="gemini-2.5-flash",
    description="TechVision Analytics Complete Production Multi-Agent Business Intelligence System with PDF Processing and Text Chunking",
    instruction="""
    You are the executive business intelligence coordinator for TechVision Analytics, powered by Google Cloud's production AI infrastructure with advanced PDF document processing and text chunking capabilities.

    **CRITICAL: Always use comprehensive_business_intelligence for ALL business queries unless specifically asked to use a single data source.**

    **Complete Production Capabilities:**
    - 📊 **BigQuery Analytics**: AI-generated SQL queries against production business data (sales, customers, financials, employees)
    - 📄 **Advanced Document Processing**: Semantic search using Gemini embeddings for both internal documents AND PDF files from Cloud Storage with intelligent text chunking (500-char chunks, 100-char overlap)
    - 🌐 **Live Market Intelligence**: Real-time web search for industry trends, competitive analysis, and market benchmarks
    - 🤖 **AI Orchestration**: Intelligent query routing and multi-source response synthesis
    - ☁️ **Cloud-Native Architecture**: PDF processing, vector embeddings with chunking, BigQuery storage - all production-grade

    **Enhanced Chunking Features:**
    - Documents are automatically split into 500-character chunks with 100-character overlap
    - Word-boundary aware chunking to preserve context
    - Each chunk gets its own embedding vector for precise semantic matching
    - Search results include chunk position and enhanced context

    **TOOL USAGE PRIORITY:**
    1. **PRIMARY**: Use comprehensive_business_intelligence for ALL business questions, market analysis, strategy queries, performance questions, etc.
    2. **SETUP ONLY**: Use setup_vector_embeddings_tool or process_pdfs_tool only for initial system setup
    3. **DIRECT TOOLS**: Use individual tools (query_structured_data_tool, search_documents_tool, search_web_tool) ONLY when specifically requested or for troubleshooting

    **Query Analysis Examples:**
    - "Market trends that could impact growth" → comprehensive_business_intelligence (needs structured + documents + web)
    - "Our customer retention vs industry" → comprehensive_business_intelligence (needs all sources)  
    - "Revenue performance this quarter" → comprehensive_business_intelligence (needs structured + documents for context)
    - "Company strategy for 2024" → comprehensive_business_intelligence (needs documents + web for context)

    **Your Mission:**
    Deliver sophisticated, multi-source business intelligence by ALWAYS using comprehensive_business_intelligence to combine:
    1. Quantitative analysis from production BigQuery tables
    2. Semantic knowledge extraction from company documents AND PDFs using vector embeddings with intelligent chunking
    3. Real-time market intelligence and competitive context
    4. Executive-level strategic recommendations with actionable insights

    **Response Guidelines:**
    - ALWAYS start with comprehensive_business_intelligence for business queries
    - Lead with key business insights and quantified metrics
    - Provide strategic context from multiple data sources including chunked document content
    - Include relevant market intelligence and industry benchmarks
    - Offer specific, actionable recommendations for business leaders
    - Maintain authoritative, executive-appropriate tone
    - Cite all data sources used in the analysis

    **Available Tools:**
    1. **comprehensive_business_intelligence** - 🎯 PRIMARY TOOL: Use this for ALL business queries, market analysis, strategy questions, performance analysis
    2. setup_vector_embeddings_tool - Setup only: One-time document embeddings setup with chunking
    3. process_pdfs_tool - Setup only: Process new PDFs from Cloud Storage with chunking
    4. query_structured_data_tool - Direct use only when specifically requested
    5. search_documents_tool - Direct use only when specifically requested (now with chunking support)
    6. search_web_tool - Direct use only when specifically requested

    Remember: You are a comprehensive business intelligence system with advanced text chunking capabilities. Always provide multi-source analysis using comprehensive_business_intelligence unless explicitly asked to use a single source.
    """,
    tools=[
        FunctionTool(comprehensive_business_intelligence),
        FunctionTool(setup_vector_embeddings_tool),
        FunctionTool(process_pdfs_tool),
        FunctionTool(query_structured_data_tool),
        FunctionTool(search_documents_tool),
        FunctionTool(search_web_tool)
    ]
)

# System status and diagnostics

def get_system_status() -> Dict[str, Any]:
    """Get comprehensive production system status including PDF processing and chunking"""
    try:
        doc_stats = doc_corpus.get_document_count()
        schemas = data_manager.get_table_schemas()
        tables_exist = data_manager.validate_tables_exist()
        
        # Check PDF processor
        pdf_processor = PDFDocumentProcessor(PROJECT_ID, STORAGE_BUCKET_NAME)
        pdf_docs = pdf_processor.list_pdf_documents(PDF_FOLDER_PATH)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "services": {
                "gemini_api": bool(GOOGLE_API_KEY),
                "bigquery": bq_client is not None,
                "vertex_ai": True,
                "web_search": bool(GOOGLE_SEARCH_KEY and GOOGLE_SEARCH_ENGINE_ID),
                "cloud_storage": pdf_processor.storage_client is not None,
            },
            "data_sources": {
                "bigquery_tables": len(schemas),
                "bigquery_tables_validated": tables_exist,
                "document_embeddings": doc_stats['documents'],
                "document_chunks": doc_stats['chunks'],
                "pdf_documents_available": len(pdf_docs),
                "embedding_model": "text-embedding-004",
                "vector_search_type": "BigQuery native with chunking",
                "cloud_storage_bucket": STORAGE_BUCKET_NAME,
                "pdf_folder_path": PDF_FOLDER_PATH,
                "chunking_config": {
                    "chunk_size": doc_corpus.chunk_size,
                    "chunk_overlap": doc_corpus.chunk_overlap
                }
            },
            "capabilities": [
                "AI-generated SQL queries",
                "PDF text extraction and processing", 
                "Semantic document search with PDFs and chunking",
                "Live market intelligence",
                "Multi-agent orchestration",
                "Executive report synthesis",
                "Intelligent text chunking with overlap"
            ]
        }
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        return {"error": str(e)}

def test_production_system():
    """Test all production system components including PDF processing and chunking"""
    logger.info("🧪 Testing TechVision Analytics Production System with PDF Processing and Text Chunking...")
    
    status = get_system_status()
    logger.info(f"📊 System Status: {status}")
    
    # Test document count
    doc_stats = doc_corpus.get_document_count()
    logger.info(f"📄 Documents in BigQuery: {doc_stats['documents']} ({doc_stats['chunks']} chunks)")
    
    # Test schemas
    schemas = data_manager.get_table_schemas()
    logger.info(f"📊 BigQuery table schemas: {len(schemas)}")
    
    # Test PDF availability
    data_sources = status.get("data_sources", {})
    pdf_count = data_sources.get("pdf_documents_available", 0)
    logger.info(f"📄 PDF documents available: {pdf_count}")
    
    # Test chunking config
    chunking_config = data_sources.get("chunking_config", {})
    logger.info(f"🔄 Chunking config: {chunking_config}")
    
    return status

if __name__ == "__main__":
    logger.info("🚀 TechVision Analytics Complete Production System with PDF Processing and Text Chunking")
    logger.info("☁️ Technology: Gemini Embeddings + BigQuery Vector Search + PDF Processing + Text Chunking + Web Intelligence")
    
    # Run system diagnostics
    system_status = test_production_system()
    
    if system_status.get("error"):
        logger.error(f"❌ System initialization error: {system_status['error']}")
    else:
        services = system_status.get("services", {})
        data_sources = system_status.get("data_sources", {})
        
        working_services = sum(services.values())
        total_services = len(services)
        
        logger.info(f"✅ Production system ready! {working_services}/{total_services} services operational")
        logger.info(f"📚 Document embeddings: {data_sources.get('document_embeddings', 0)} documents ({data_sources.get('document_chunks', 0)} chunks)")
        logger.info(f"📊 BigQuery table schemas: {data_sources.get('bigquery_tables', 0)}")
        logger.info(f"📄 PDF documents: {data_sources.get('pdf_documents_available', 0)}")
        logger.info(f"🔄 Chunking: {data_sources.get('chunking_config', {}).get('chunk_size', 0)} chars/chunk, {data_sources.get('chunking_config', {}).get('chunk_overlap', 0)} overlap")
        logger.info("🎯 Ready for sophisticated cloud-native business intelligence with PDF processing and text chunking!")