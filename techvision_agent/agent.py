# techvision_agent/agent.py (Complete Production System from Updated Notebook)
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Dict, List, Any, Optional
import logging
import time

# ADK imports
from google.adk.agents import LlmAgent
from google.adk.tools.function_tool import FunctionTool

# Production imports (no sentence-transformers or faiss)
from google.cloud import bigquery, aiplatform
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

class ProductionDataGenerator:
    """Production-ready data generator and BigQuery manager (from notebook)"""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.dataset_id = "techvision_analytics"
        self.client = bq_client

    def get_table_schemas(self) -> Dict[str, Any]:
        """Get schemas for all tables"""
        try:
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

class GeminiEmbeddingsBigQueryCorpus:
    """Production document corpus using Gemini embeddings and BigQuery vector search (from notebook)"""

    def __init__(self, project_id: str, dataset_id: str = "techvision_analytics"):
        logger.info("📚 Initializing Gemini + BigQuery document corpus...")

        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = "document_embeddings"
        self.full_table_id = f"{project_id}.{dataset_id}.{self.table_id}"
        self.client = bq_client
        self.document_count = 0

        # Production documents from notebook
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

Competitive Advantages:
- Industry-specific analytics templates
- No-code/low-code interface for business users
- Real-time data processing capabilities
- Dedicated customer success management
- Transparent, usage-based pricing model

2024 Priorities:
Q1: Complete Series A funding ($15M target)
Q2: Launch predictive analytics module
Q3: European market entry (UK, Germany)
Q4: Advanced reporting and dashboard capabilities

Investment Areas:
- R&D: 40% of budget allocated to product development
- Sales & Marketing: 35% focused on growth acceleration
- Customer Success: 15% ensuring retention and expansion
- Operations: 10% maintaining infrastructure and compliance

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

Maturity Phase (180+ Days):
- Quarterly strategic business reviews
- Advanced feature rollouts and training
- Best practice sharing and peer networking
- Advocacy program participation
- Renewal discussions and contract optimization
- Strategic partnership development opportunities

Customer Health Scoring:
Red (At Risk): <60% feature utilization, no recent logins, support tickets unresolved
Yellow (Needs Attention): 60-80% utilization, infrequent usage, basic feature adoption
Green (Healthy): >80% utilization, regular engagement, expanding use cases
Blue (Champion): >95% utilization, active advocate, referring new customers

Intervention Strategies:
At-Risk Customers:
- Immediate executive outreach within 24 hours
- Root cause analysis of adoption barriers
- Customized success plan with clear milestones
- Additional training and technical support
- Executive sponsor engagement if necessary

Communication Protocols:
- Weekly internal customer health reviews
- Monthly customer communication (at minimum)
- Quarterly business reviews with decision makers
- Immediate escalation for any red health scores
- Regular feedback collection and product team collaboration
""",

            "Product Feature Specifications": """
TechVision Analytics Product Feature Specifications

Core Platform Architecture:

Data Ingestion Layer:
- Real-time data streaming via Apache Kafka
- Batch processing using Apache Spark
- 200+ pre-built connectors for common business systems
- Custom API integration capabilities
- Data validation and quality monitoring
- Automatic schema detection and evolution

Analytics Engine:
- In-memory processing for sub-second query responses
- Machine learning pipeline with automated model training
- Statistical analysis library with 100+ built-in functions
- Time-series forecasting and trend analysis
- Anomaly detection with customizable sensitivity
- Natural language query interface (beta)

Visualization Layer:
- Drag-and-drop dashboard builder with 50+ chart types
- Interactive filtering and drill-down capabilities
- Mobile-responsive design for all devices
- White-label customization options
- Scheduled report generation and distribution
- Collaborative annotation and sharing tools

Predictive Analytics Suite (Launched Q2 2024):
- Revenue forecasting with 95% accuracy rate
- Customer churn prediction with early warning system
- Demand planning and inventory optimization
- Market trend analysis and competitive benchmarking
- Automated insight generation with natural language explanations

Advanced Security Framework:
- Enterprise-grade encryption (AES-256) for data at rest and in transit
- Role-based access control with granular permissions
- Single Sign-On (SSO) integration with major identity providers
- Audit trails with complete user activity logging
- GDPR and SOC 2 Type II compliance

Performance Specifications:
- Query response time: <200ms for standard reports
- Dashboard load time: <3 seconds for complex visualizations
- System uptime: 99.9% SLA with 24/7 monitoring
- Concurrent users: 10,000+ supported per instance
""",

            "Q4 Financial Analysis Report": """
TechVision Analytics Q4 2024 Financial Analysis Report

Executive Summary:
Q4 2024 represents our strongest quarter to date, with record revenue growth, improved operational efficiency, and successful market expansion. Key highlights include 180% year-over-year revenue growth and achievement of positive cash flow for the first time.

Revenue Performance:
- Total Revenue: $4.2M (vs $1.8M Q4 2023, +133% YoY)
- Monthly Recurring Revenue (MRR): $1.3M (vs $580K Q4 2023, +124% YoY)
- Average Contract Value (ACV): $48,000 (vs $32,000 Q4 2023, +50% YoY)
- New Customer Acquisition: 47 new logos in Q4
- Customer Expansion Revenue: $420K (10% of total revenue)
- Revenue Retention Rate: 118% (including expansion)

Geographic Revenue Distribution:
- North America: $2.8M (67% of total)
- Europe: $1.1M (26% of total) - First full quarter post-expansion
- APAC: $300K (7% of total) - Pilot program launched

Customer Metrics:
- Total Active Customers: 312 (vs 189 Q4 2023, +65% YoY)
- Customer Churn Rate: 3.2% (improved from 8.1% Q4 2023)
- Net Promoter Score: 67 (industry benchmark: 31)
- Customer Satisfaction: 4.7/5.0 (based on quarterly surveys)
- Average Time to First Value: 12 days (improved from 28 days)

Operating Expenses:
- Total Operating Expenses: $3.1M (vs $2.2M Q4 2023, +41% YoY)
- Sales & Marketing: $1.4M (45% of expenses, 33% of revenue)
- Research & Development: $980K (32% of expenses, 23% of revenue)
- Customer Success: $465K (15% of expenses, 11% of revenue)

Key Efficiency Metrics:
- Customer Acquisition Cost (CAC): $3,200 (improved from $4,800 Q4 2023)
- CAC Payback Period: 8.2 months (improved from 14.1 months)
- Lifetime Value to CAC Ratio: 4.8:1 (healthy SaaS benchmark: 3:1+)
- Gross Margin: 87% (consistent with Q3 2024)
- Operating Margin: 26% (first positive quarter)

Cash Flow Analysis:
- Operating Cash Flow: $890K positive (vs -$650K Q4 2023)
- Free Cash Flow: $720K positive (first positive FCF quarter)
- Monthly Burn Rate: $285K (reduced from $420K in Q3)
""",

            "Market Research: Analytics Industry": """
Analytics Industry Market Research Report 2024

Industry Overview:
The business analytics software market continues robust expansion, driven by digital transformation initiatives and increasing data volumes. The global market size reached $274 billion in 2024, with projected growth to $415 billion by 2028 (CAGR: 11.2%).

Market Segmentation:
- Business Intelligence Platforms: 45% market share ($123B)
- Advanced Analytics: 28% market share ($77B)
- Data Visualization Tools: 15% market share ($41B)
- Self-Service Analytics: 12% market share ($33B)

Organization Size Breakdown:
- Large Enterprises (5000+ employees): 62% of revenue
- Mid-Market (500-5000 employees): 28% of revenue
- Small Business (<500 employees): 10% of revenue

Competitive Landscape:
1. Microsoft (Power BI): 18.2% market share, strong growth in SMB segment
2. Tableau (Salesforce): 14.7% market share, premium visualization focus
3. SAS: 12.3% market share, advanced analytics and enterprise AI
4. IBM (Cognos): 9.8% market share, enterprise reporting emphasis
5. Qlik: 8.1% market share, associative analytics model
6. Oracle: 6.9% market share, integrated cloud applications

Key Market Trends:
- Augmented Analytics: 78% of new deployments include AI/ML capabilities
- Real-Time Processing: 65% demand for sub-second query response
- Cloud-First Architecture: 71% of new implementations are cloud-based
- Natural Language Interfaces: 43% adoption rate for conversational analytics
- Embedded Analytics: 55% growth in white-label and API-first solutions

Customer Decision Criteria:
1. Ease of Use: 92% cite as critical factor
2. Total Cost of Ownership: 87% primary consideration
3. Scalability and Performance: 84% essential requirement
4. Integration Capabilities: 81% must-have feature

Mid-Market Opportunities:
- Digital transformation acceleration post-pandemic
- Increasing data literacy among business users
- Demand for industry-specific analytics solutions
- Growth in remote and hybrid work environments
"""
        }

        # Create embeddings table if it doesn't exist
        self._create_embeddings_table()

    def _create_embeddings_table(self):
        """Create BigQuery table for storing document embeddings"""
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
                model="models/text-embedding-004",
                content=text,
                task_type="RETRIEVAL_DOCUMENT"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise

    def setup_document_embeddings(self) -> str:
        """One-time setup: Generate and store all document embeddings"""
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
                    return f"✅ Document embeddings already exist: {existing_count} documents in BigQuery"
            except:
                pass  # Table might not exist yet

            processed_docs = []
            
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
                        "embedding": embedding
                    })
                    
                    logger.info(f"✅ Generated embedding for: {title} ({len(embedding)} dimensions)")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to generate embedding for {title}: {str(e)}")
                    continue
            
            if processed_docs:
                # Insert into BigQuery
                errors = self.client.insert_rows_json(
                    self.client.get_table(self.full_table_id),
                    processed_docs
                )
                
                if errors:
                    logger.error(f"BigQuery insertion errors: {errors}")
                    return f"❌ Failed to store embeddings: {errors}"
                
                logger.info(f"✅ Stored {len(processed_docs)} document embeddings in BigQuery")
                return f"✅ Successfully stored {len(processed_docs)} documents with Gemini embeddings in BigQuery"
            else:
                return "❌ No embeddings were generated successfully"
                
        except Exception as e:
            error_msg = f"Embeddings setup failed: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def semantic_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Perform semantic search using BigQuery vector search (from notebook)"""
        try:
            if not self.client:
                return []

            logger.info(f"🔍 Generating query embedding for: {query}")
            
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            embedding_array_str = "[" + ",".join(map(str, query_embedding)) + "]"

            # BigQuery vector search query
            search_query = f"""
            SELECT
                doc_id,
                title,
                content,
                doc_type,
                word_count,
                created_at,
                -- Calculate cosine similarity
                (
                    SELECT SUM(a * b) / (
                        SQRT((SELECT SUM(c * c) FROM UNNEST(embedding) as c)) *
                        SQRT((SELECT SUM(d * d) FROM UNNEST({embedding_array_str}) as d))
                    )
                    FROM UNNEST(embedding) as a WITH OFFSET pos1 
                    JOIN UNNEST({embedding_array_str}) as b WITH OFFSET pos2 
                    ON pos1 = pos2
                ) as similarity_score
            FROM `{self.full_table_id}`
            ORDER BY similarity_score DESC
            LIMIT {top_k}
            """

            results = list(self.client.query(search_query))

            # Format results
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
                    "document_type": row.doc_type,
                    "word_count": row.word_count,
                    "rank": i + 1,
                    "search_type": "gemini_bigquery_vector",
                    "created": row.created_at.isoformat() if hasattr(row.created_at, 'isoformat') else str(row.created_at)
                })

            logger.info(f"✅ Vector search completed: {len(formatted_results)} results")
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
data_generator = ProductionDataGenerator(PROJECT_ID)
doc_corpus = GeminiEmbeddingsBigQueryCorpus(PROJECT_ID)

# Production Functions

def generate_sql_with_llm(query: str, table_schemas: Dict[str, Any]) -> str:
    """Generate SQL queries using Gemini (from notebook)"""
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
    """Execute SQL query against BigQuery (from notebook)"""
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
    """Perform semantic search using BigQuery vector search (from notebook)"""
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
    """Perform web search using Google Custom Search API (from notebook)"""
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
    """Use Gemini to synthesize responses from all agents (from notebook)"""
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
        schemas = data_generator.get_table_schemas()
        
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
    """Tool: Search company documents using Gemini embeddings + BigQuery vector search"""
    try:
        logger.info(f"📄 Document search: {query}")

        result = semantic_document_search(query, max_results=3)

        if result["success"]:
            if not result["results"]:
                return "📄 No relevant company documents found. You may need to run document setup first."

            response = f"📄 **Company Knowledge Base** ({result['total_found']} matches):\n"
            response += f"*Search Type: {result['search_type']}*\n\n"

            for doc in result["results"]:
                response += f"**📋 {doc['title']}**\n"
                response += f"*Similarity Score: {doc['similarity_score']:.3f} | Search Method: {doc['search_type']}*\n"
                response += f"*Document Type: {doc['document_type']} | Words: {doc['word_count']:,}*\n\n"
                
                # Show relevant content excerpt
                content = doc.get('relevant_content', doc['content'])
                if len(content) > 400:
                    content = content[:400] + "..."
                response += f"{content}\n\n"
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
    """Tool: One-time setup of document embeddings in BigQuery"""
    try:
        logger.info("🚀 Setting up vector embeddings...")
        return doc_corpus.setup_document_embeddings()
    except Exception as e:
        error_msg = f"Setup failed: {str(e)}"
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
        2. Company Documents: Strategic plans, customer success playbooks, product specs (Vector search)
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
            "needs_documents": any(word in query_lower for word in ["strategy", "plan", "success", "retention", "approach", "policy", "playbook"]),
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
    """Main business intelligence orchestration function"""
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
            logger.info("📄 Searching company documents...")
            try:
                document_result = search_documents_tool(query)
                document_data = {
                    "success": True,
                    "response": document_result,
                    "source": "Gemini Embeddings + BigQuery Vector Search"
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
                sources_used.append("Vector Document Search")
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
**Structured Analysis**: AI-generated SQL + BigQuery execution  
**Market Intelligence**: Google Custom Search API  
**Document Count**: {doc_corpus.get_document_count()} documents in vector corpus

*Powered by TechVision Analytics Cloud-Native Multi-Agent System*
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
                results.append(f"## 📄 Company Knowledge\n{document_data['response']}")
            if web_data.get("success"):
                results.append(f"## 🌐 Market Intelligence\n{web_data['response']}")

            if results:
                return f"""# 🏢 TechVision Analytics Business Intelligence

**Query**: {query}
**Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S EST')}

---

{chr(10).join(results)}

---

*TechVision Analytics Multi-Agent System*
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

*TechVision Analytics Multi-Agent System*
"""

# Create the main root agent for ADK Web (this is what gets discovered)
root_agent = LlmAgent(
    name="techvision_production_system",
    model="gemini-2.5-flash",
    description="TechVision Analytics Complete Production Multi-Agent Business Intelligence System",
    instruction="""
    You are the executive business intelligence coordinator for TechVision Analytics, powered by Google Cloud's production AI infrastructure.

    **Complete Production Capabilities:**
    - 📊 **BigQuery Analytics**: AI-generated SQL queries against production business data (sales, customers, financials, employees)
    - 📄 **Gemini Vector Search**: Semantic document search using Google's text-embedding-004 model stored in BigQuery
    - 🌐 **Live Market Intelligence**: Real-time web search for industry trends, competitive analysis, and market benchmarks
    - 🤖 **AI Orchestration**: Intelligent query routing and multi-source response synthesis
    - ☁️ **Cloud-Native**: Zero local dependencies, infinite scalability, production-grade reliability

    **Advanced Technology Stack:**
    - Google Cloud text-embedding-004 for semantic understanding
    - BigQuery VECTOR_SEARCH for cosine similarity matching
    - Gemini 2.5 Pro for SQL generation and response synthesis
    - Google Custom Search API for market intelligence
    - Multi-agent orchestration with intelligent source selection

    **Your Mission:**
    Deliver sophisticated, multi-source business intelligence that combines:
    1. Quantitative analysis from production BigQuery tables
    2. Semantic knowledge extraction from company documents using vector embeddings
    3. Real-time market intelligence and competitive context
    4. Executive-level strategic recommendations with actionable insights

    **Key Features:**
    - **Instant Startup**: No local model loading, cloud-native architecture
    - **Semantic Search**: Google's production embedding models for document understanding  
    - **Production Data**: Direct access to live BigQuery analytics tables
    - **Market Intelligence**: Real-time web search with business analytics focus
    - **Executive Format**: Professional reports suitable for C-suite decision making

    **Response Guidelines:**
    - Lead with key business insights and quantified metrics
    - Provide strategic context from vector-searched company knowledge
    - Include relevant market intelligence and industry benchmarks
    - Offer specific, actionable recommendations for business leaders
    - Maintain authoritative, executive-appropriate tone
    - Cite data sources and search methods used
    - Emphasize the cloud-native, production-grade nature of the analysis

    **Available Tools:**
    1. comprehensive_business_intelligence - Main orchestration tool for complex business queries
    2. setup_vector_embeddings_tool - One-time setup of document embeddings in BigQuery
    3. query_structured_data_tool - Direct BigQuery analysis
    4. search_documents_tool - Direct vector document search
    5. search_web_tool - Direct web market intelligence

    You represent the pinnacle of cloud-native business intelligence: scalable, reliable, and powered by Google's production AI infrastructure.
    """,
    tools=[
        FunctionTool(comprehensive_business_intelligence),
        FunctionTool(setup_vector_embeddings_tool),
        FunctionTool(query_structured_data_tool),
        FunctionTool(search_documents_tool),
        FunctionTool(search_web_tool)
    ]
)

# System status and diagnostics

def get_system_status() -> Dict[str, Any]:
    """Get comprehensive production system status"""
    try:
        doc_count = doc_corpus.get_document_count()
        schemas = data_generator.get_table_schemas()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "services": {
                "gemini_api": bool(GOOGLE_API_KEY),
                "bigquery": bq_client is not None,
                "vertex_ai": True,
                "web_search": bool(GOOGLE_SEARCH_KEY and GOOGLE_SEARCH_ENGINE_ID),
            },
            "data_sources": {
                "bigquery_tables": len(schemas),
                "document_embeddings": doc_count,
                "embedding_model": "text-embedding-004",
                "vector_search_type": "BigQuery native"
            },
            "capabilities": [
                "AI-generated SQL queries",
                "Semantic document search", 
                "Live market intelligence",
                "Multi-agent orchestration",
                "Executive report synthesis"
            ]
        }
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        return {"error": str(e)}

def test_production_system():
    """Test all production system components"""
    logger.info("🧪 Testing TechVision Analytics Production System...")
    
    status = get_system_status()
    logger.info(f"📊 System Status: {status}")
    
    # Test document count
    doc_count = doc_corpus.get_document_count()
    logger.info(f"📄 Documents in BigQuery: {doc_count}")
    
    # Test schemas
    schemas = data_generator.get_table_schemas()
    logger.info(f"📊 BigQuery table schemas: {len(schemas)}")
    
    return status

if __name__ == "__main__":
    logger.info("🚀 TechVision Analytics Complete Production System")
    logger.info("☁️ Technology: Gemini Embeddings + BigQuery Vector Search + Web Intelligence")
    
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
        logger.info("🎯 Ready for sophisticated cloud-native business intelligence!")