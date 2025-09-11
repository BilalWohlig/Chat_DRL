# techvision_agent/agent.py (Complete Production System with PDF Processing)
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
    """Production document corpus using Gemini embeddings and BigQuery vector search with PDF integration"""

    def __init__(self, project_id: str, dataset_id: str = "techvision_analytics"):
        logger.info("📚 Initializing Gemini + BigQuery document corpus with PDF processing...")

        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = "document_embeddings_new"
        self.full_table_id = f"{project_id}.{dataset_id}.{self.table_id}"
        self.client = bq_client
        self.document_count = 0

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
        """Create BigQuery table for storing document embeddings with PDF support"""
        if not self.client:
            logger.error("BigQuery client not available")
            return

        try:
            schema = [
                bigquery.SchemaField("doc_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("title", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("content", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("doc_type", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("word_count", "INTEGER", mode="REQUIRED"),
                bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
                bigquery.SchemaField("source_blob", "STRING", mode="NULLABLE"),  # New field for PDF source
            ]

            table = bigquery.Table(self.full_table_id, schema=schema)
            table = self.client.create_table(table, exists_ok=True)
            logger.info(f"✅ Document embeddings table ready: {self.full_table_id}")

        except Exception as e:
            logger.error(f"Failed to create embeddings table: {str(e)}")

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
        """Add PDF document to BigQuery with Gemini embeddings"""
        try:
            if not self.client:
                raise Exception("BigQuery client not available")

            # Generate unique doc ID
            doc_id = f"PDF_{self.document_count:03d}"
            self.document_count += 1

            # Generate embedding using Gemini
            logger.info(f"🧠 Generating Gemini embedding for: {pdf_data['title'][:50]}...")
            embedding = self._generate_embedding(pdf_data['content'])

            # Prepare row data with proper datetime formatting
            row_data = {
                "doc_id": doc_id,
                "title": pdf_data['title'],
                "content": pdf_data['content'],
                "doc_type": "pdf",
                "word_count": pdf_data['word_count'],
                "created_at": datetime.now().isoformat(),
                "embedding": embedding,
                "source_blob": pdf_data.get('blob_name', '')
            }

            # Insert into BigQuery
            errors = self.client.insert_rows_json(
                self.client.get_table(self.full_table_id),
                [row_data]
            )

            if errors:
                raise Exception(f"BigQuery insertion errors: {errors}")

            logger.info(f"✅ Added PDF document: {pdf_data['title']} (ID: {doc_id}, {len(embedding)} dimensions)")

        except Exception as e:
            raise Exception(f"Failed to add PDF document: {str(e)}")

    def process_all_pdfs(self, folder_path: str = PDF_FOLDER_PATH):
        """Process all PDFs from Cloud Storage and add to corpus"""
        try:
            # Initialize PDF processor
            pdf_processor = PDFDocumentProcessor(PROJECT_ID, STORAGE_BUCKET_NAME)
            
            # Get list of PDF documents
            pdf_documents = pdf_processor.list_pdf_documents(folder_path)

            if not pdf_documents:
                logger.warning("No PDF documents found in the specified folder")
                return "⚠️ No PDF documents found"

            logger.info(f"📄 Processing {len(pdf_documents)} PDF documents...")

            processed_count = 0
            for i, pdf_info in enumerate(pdf_documents):
                try:
                    logger.info(f"📖 Processing PDF {i+1}/{len(pdf_documents)}: {pdf_info['title']}")

                    # Download and extract text
                    pdf_data = pdf_processor.download_and_extract_text(pdf_info['blob_name'])

                    # Add to corpus with embeddings
                    self.add_pdf_document(pdf_data)
                    processed_count += 1

                except Exception as e:
                    logger.error(f"Failed to process {pdf_info['title']}: {str(e)}")
                    continue

            return f"✅ Successfully processed {processed_count}/{len(pdf_documents)} PDF documents"

        except Exception as e:
            error_msg = f"Failed to process PDFs: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def setup_document_embeddings(self) -> str:
        """One-time setup: Generate and store all document embeddings including PDFs"""
        try:
            if not self.client:
                return "❌ BigQuery client not available"

            logger.info("🚀 Starting document embeddings setup...")
            
            # Check if documents already exist
            try:
                count_query = f"SELECT COUNT(*) as total FROM `{self.full_table_id}`"
                result = list(self.client.query(count_query))
                existing_count = result[0].total if result else 0
                
                if existing_count > 0:
                    # Also process PDFs if they exist
                    pdf_result = self.process_all_pdfs()
                    return f"✅ Document embeddings already exist: {existing_count} documents. {pdf_result}"
            except:
                pass  # Table might not exist yet

            processed_docs = []
            
            # Process built-in documents
            for doc_id, (title, content) in enumerate(self.production_documents.items()):
                try:
                    logger.info(f"🧠 Generating Gemini embedding for: {title}")
                    
                    # Generate embedding using Gemini
                    embedding = self._generate_embedding(content)
                    
                    processed_docs.append({
                        "doc_id": f"doc_{doc_id:03d}",
                        "title": title,
                        "content": content,
                        "doc_type": "internal",
                        "word_count": len(content.split()),
                        "created_at": datetime.now(),
                        "embedding": embedding,
                        "source_blob": None
                    })
                    
                    logger.info(f"✅ Generated embedding for: {title} ({len(embedding)} dimensions)")
                    
                except Exception as e:
                    logger.error(f"Failed to generate embedding for {title}: {str(e)}")
                    continue
            
            response_messages = []
            
            if processed_docs:
                # Insert into BigQuery
                errors = self.client.insert_rows_json(
                    self.client.get_table(self.full_table_id),
                    processed_docs
                )
                
                if errors:
                    logger.error(f"BigQuery insertion errors: {errors}")
                    response_messages.append(f"❌ Failed to store embeddings: {errors}")
                else:
                    logger.info(f"✅ Stored {len(processed_docs)} document embeddings in BigQuery")
                    response_messages.append(f"✅ Successfully stored {len(processed_docs)} documents with Gemini embeddings")
            
            # Also process PDFs
            pdf_result = self.process_all_pdfs()
            response_messages.append(pdf_result)
            
            return " | ".join(response_messages) if response_messages else "❌ No embeddings were generated successfully"
                
        except Exception as e:
            error_msg = f"Embeddings setup failed: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def semantic_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Working BigQuery VECTOR_SEARCH without JOIN - access via base struct"""
        try:
            if not self.client:
                return []

            # Generate query embedding
            logger.info(f"🔍 Generating query embedding for: {query[:50]}...")
            query_embedding = self._generate_embedding(query)
            embedding_array_str = "[" + ",".join(map(str, query_embedding)) + "]"

            # METHOD 1: Try direct VECTOR_SEARCH (recommended)
            try:
                search_query = f"""
                SELECT
                    base.doc_id,
                    base.title,
                    base.content,
                    base.doc_type,
                    base.word_count,
                    base.created_at,
                    base.source_blob,
                    distance,
                    (1 - distance) AS similarity_score
                FROM VECTOR_SEARCH(
                    TABLE `{self.full_table_id}`,
                    'embedding',
                    (SELECT {embedding_array_str} AS embedding),
                    distance_type => 'COSINE',
                    top_k => {top_k}
                )
                ORDER BY distance ASC
                """

                logger.info("🔎 Attempting direct VECTOR_SEARCH...")
                results = list(self.client.query(search_query))
                logger.info(f"✅ Direct VECTOR_SEARCH successful!")

            except Exception as direct_error:
                logger.warning(f"Direct approach failed: {str(direct_error)[:100]}")

                # METHOD 2: Fallback to table-to-table approach
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
                        base.title,
                        base.content,
                        base.doc_type,
                        base.word_count,
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
                        top_k => {top_k}
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

            # Format results consistently
            formatted_results = []
            for i, row in enumerate(results):
                content = row.content
                sentences = [s.strip() for s in content.split('.') if s.strip()]
                relevant_content = '. '.join(sentences[:5]) + '...' if len(sentences) > 5 else content

                formatted_results.append({
                    "id": row.doc_id,
                    "title": row.title,
                    "content": row.content,
                    "relevant_content": relevant_content,
                    "similarity_score": float(row.similarity_score),
                    "distance": float(row.distance),
                    "document_type": row.doc_type,
                    "word_count": row.word_count,
                    "rank": i + 1,
                    "created": row.created_at.isoformat() if hasattr(row.created_at, 'isoformat') else str(row.created_at),
                    "source_blob": row.source_blob if hasattr(row, 'source_blob') else None
                })

            logger.info(f"✅ Found {len(formatted_results)} similar documents")
            return formatted_results

        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            return []

    def get_document_count(self) -> int:
        """Get total number of documents in corpus"""
        try:
            if not self.client:
                return 0
            count_query = f"SELECT COUNT(*) as total FROM `{self.full_table_id}`"
            result = list(self.client.query(count_query))
            return result[0].total if result else 0
        except Exception as e:
            logger.warning(f"Failed to get document count: {str(e)}")
            return 0

# Initialize production components
data_manager = ProductionDataManager(PROJECT_ID)
doc_corpus = GeminiEmbeddingsBigQueryCorpus(PROJECT_ID)

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
        2. Use proper BigQuery syntax
        3. Include appropriate WHERE clauses, ORDER BY, and LIMIT as needed
        4. When filtering by date columns, use PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%E*S', column_name)
        5. Handle date ranges correctly using parsed timestamps
        6. For subtracting months or years, use DATE_SUB function
        7. When comparing TIMESTAMP with DATE, cast TIMESTAMP to DATE
        8. Ensure query is safe (no DROP, DELETE, TRUNCATE operations)
        9. Use table aliases for readability
        10. Use LIMIT 10 for testing

        SQL Query:
        """

        response = model.generate_content(prompt)
        sql_query = response.text.strip()

        # Clean up the response
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]

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

def semantic_document_search(query: str, max_results: int = 3) -> Dict[str, Any]:
    """Perform semantic search using BigQuery vector search"""
    try:
        results = doc_corpus.semantic_search(query, top_k=max_results)

        return {
            "success": True,
            "query": query,
            "results": results,
            "total_found": len(results),
            "search_type": "bigquery_vector_search_with_gemini_embeddings"
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
    """Tool: Search company documents including PDFs using Gemini embeddings + BigQuery vector search"""
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
                response += f"*Words: {doc['word_count']:,}*\n"
                
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
    """Tool: One-time setup of document embeddings in BigQuery including PDF processing"""
    try:
        logger.info("🚀 Setting up vector embeddings...")
        return doc_corpus.setup_document_embeddings()
    except Exception as e:
        error_msg = f"Setup failed: {str(e)}"
        logger.error(error_msg)
        return f"❌ {error_msg}"

def process_pdfs_tool() -> str:
    """Tool: Process PDFs from Cloud Storage and add to document corpus"""
    try:
        logger.info("📄 Processing PDFs from Cloud Storage...")
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
        2. Company Documents: Strategic plans, customer success playbooks, product specs, PDFs (Vector search)
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
    """Main business intelligence orchestration function with PDF support"""
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
            logger.info("📄 Searching company documents including PDFs...")
            try:
                document_result = search_documents_tool(query)
                document_data = {
                    "success": True,
                    "response": document_result,
                    "source": "Gemini Embeddings + BigQuery Vector Search + PDF Processing"
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
                sources_used.append("Vector Document Search + PDF Processing")
            if web_data.get("success"):
                sources_used.append("Web Intelligence")

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
**Document Processing**: PDF text extraction + semantic embeddings  
**Structured Analysis**: AI-generated SQL + BigQuery execution  
**Market Intelligence**: Google Custom Search API  
**Document Count**: {doc_corpus.get_document_count()} documents in vector corpus  
**Cloud Storage**: {STORAGE_BUCKET_NAME}/{PDF_FOLDER_PATH}

*Powered by TechVision Analytics Cloud-Native Multi-Agent System with PDF Processing*
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
                results.append(f"## 📄 Company Knowledge (including PDFs)\n{document_data['response']}")
            if web_data.get("success"):
                results.append(f"## 🌐 Market Intelligence\n{web_data['response']}")

            if results:
                return f"""# 🏢 TechVision Analytics Business Intelligence

**Query**: {query}
**Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S EST')}

---

{chr(10).join(results)}

---

*TechVision Analytics Multi-Agent System with PDF Processing*
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

*TechVision Analytics Multi-Agent System with PDF Processing*
"""

# Create the main root agent for ADK Web (this is what gets discovered)
root_agent = LlmAgent(
    name="techvision_production_system_with_pdf",
    model="gemini-2.5-flash",
    description="TechVision Analytics Complete Production Multi-Agent Business Intelligence System with PDF Processing",
    instruction="""
    You are the executive business intelligence coordinator for TechVision Analytics, powered by Google Cloud's production AI infrastructure with advanced PDF document processing capabilities.

    **CRITICAL: Always use comprehensive_business_intelligence for ALL business queries unless specifically asked to use a single data source.**

    **Complete Production Capabilities:**
    - 📊 **BigQuery Analytics**: AI-generated SQL queries against production business data (sales, customers, financials, employees)
    - 📄 **Advanced Document Processing**: Semantic search using Gemini embeddings for both internal documents AND PDF files from Cloud Storage
    - 🌐 **Live Market Intelligence**: Real-time web search for industry trends, competitive analysis, and market benchmarks
    - 🤖 **AI Orchestration**: Intelligent query routing and multi-source response synthesis
    - ☁️ **Cloud-Native Architecture**: PDF processing, vector embeddings, BigQuery storage - all production-grade

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
    2. Semantic knowledge extraction from company documents AND PDFs using vector embeddings
    3. Real-time market intelligence and competitive context
    4. Executive-level strategic recommendations with actionable insights

    **Response Guidelines:**
    - ALWAYS start with comprehensive_business_intelligence for business queries
    - Lead with key business insights and quantified metrics
    - Provide strategic context from multiple data sources
    - Include relevant market intelligence and industry benchmarks
    - Offer specific, actionable recommendations for business leaders
    - Maintain authoritative, executive-appropriate tone
    - Cite all data sources used in the analysis

    **Available Tools:**
    1. **comprehensive_business_intelligence** - 🎯 PRIMARY TOOL: Use this for ALL business queries, market analysis, strategy questions, performance analysis
    2. setup_vector_embeddings_tool - Setup only: One-time document embeddings setup
    3. process_pdfs_tool - Setup only: Process new PDFs from Cloud Storage
    4. query_structured_data_tool - Direct use only when specifically requested
    5. search_documents_tool - Direct use only when specifically requested  
    6. search_web_tool - Direct use only when specifically requested

    Remember: You are a comprehensive business intelligence system. Always provide multi-source analysis using comprehensive_business_intelligence unless explicitly asked to use a single source.
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
    """Get comprehensive production system status including PDF processing"""
    try:
        doc_count = doc_corpus.get_document_count()
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
                "document_embeddings": doc_count,
                "pdf_documents_available": len(pdf_docs),
                "embedding_model": "text-embedding-004",
                "vector_search_type": "BigQuery native",
                "cloud_storage_bucket": STORAGE_BUCKET_NAME,
                "pdf_folder_path": PDF_FOLDER_PATH
            },
            "capabilities": [
                "AI-generated SQL queries",
                "PDF text extraction and processing", 
                "Semantic document search with PDFs",
                "Live market intelligence",
                "Multi-agent orchestration",
                "Executive report synthesis"
            ]
        }
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        return {"error": str(e)}

def test_production_system():
    """Test all production system components including PDF processing"""
    logger.info("🧪 Testing TechVision Analytics Production System with PDF Processing...")
    
    status = get_system_status()
    logger.info(f"📊 System Status: {status}")
    
    # Test document count
    doc_count = doc_corpus.get_document_count()
    logger.info(f"📄 Documents in BigQuery: {doc_count}")
    
    # Test schemas
    schemas = data_manager.get_table_schemas()
    logger.info(f"📊 BigQuery table schemas: {len(schemas)}")
    
    # Test PDF availability
    data_sources = status.get("data_sources", {})
    pdf_count = data_sources.get("pdf_documents_available", 0)
    logger.info(f"📄 PDF documents available: {pdf_count}")
    
    return status

if __name__ == "__main__":
    logger.info("🚀 TechVision Analytics Complete Production System with PDF Processing")
    logger.info("☁️ Technology: Gemini Embeddings + BigQuery Vector Search + PDF Processing + Web Intelligence")
    
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
        logger.info(f"📚 Document embeddings: {data_sources.get('document_embeddings', 0)}")
        logger.info(f"📊 BigQuery tables: {data_sources.get('bigquery_tables', 0)}")
        logger.info(f"📄 PDF documents: {data_sources.get('pdf_documents_available', 0)}")
        logger.info("🎯 Ready for sophisticated cloud-native business intelligence with PDF processing!")