# Multi-LLM Chat with MCP Integration

A comprehensive chat system that combines multiple language models (OpenAI GPT, Cohere Command) with Model Context Protocol (MCP) tool integration, advanced database migration capabilities, and containerized infrastructure for scalable AI applications.

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Chat Interface│    │ MCP Aggregator  │    │ MCP Servers     │
│   (Streamlit)   │◄──►│   (FastMCP)     │◄──►│ (stdio/HTTP)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                                            │
         ▼                                            ▼
┌─────────────────┐                           ┌─────────────────┐
│ LLM Providers   │                           │ Tool Execution  │
│ OpenAI/Cohere   │                           │ & Logging       │
└─────────────────┘                           └─────────────────┘
                                                      │
                                                      ▼
                          ┌────────────────────────────────────────────────────────────┐
                          │                Database Infrastructure                     │
                          │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
                          │  │    MySQL    │  │    Neo4j    │  │Elasticsearch│         │
                          │  │   (Source)  │  │  (Graph DB) │  │ (Search DB) │         │
                          │  └─────────────┘  └─────────────┘  └─────────────┘         │
                          └────────────────────────────────────────────────────────────┘
                                                    ▲
                                                    │
                                            ┌─────────────────┐
                                            │ Migration Tool  │
                                            │ (MySQL → Graph/ │
                                            │    Search DB)   │
                                            └─────────────────┘
```

## 📁 Component Overview

### 1. Chat Interface (`chat/`)
**Purpose**: Web-based multi-LLM chat interface with MCP tool integration

**Key Features**:
- Support for multiple LLM providers (OpenAI GPT models, Cohere Command models)
- Dynamic MCP tool calling and execution
- Real-time streaming responses
- Token usage tracking and cost calculation
- MCP interaction logging with dedicated log viewer
- Configurable API keys and model selection

**Technology Stack**: Streamlit, OpenAI SDK, Cohere SDK, FastMCP client

### 2. MCP Aggregator (`mcp-aggregator/`)
**Purpose**: HTTP server that aggregates multiple MCP stdio servers into a single endpoint

**Key Features**:
- Dynamic tool registration from multiple MCP servers
- Health monitoring and auto-restart of backend servers
- FastMCP HTTP transport for easy integration
- Configurable server management via JSON config
- Proxy functions for tool execution routing

**Technology Stack**: FastMCP, MCP protocol, asyncio, uvicorn

### 3. MySQL Migration Tool (`mysql-migration/`)
**Purpose**: Advanced database migration from MySQL to Neo4j or Elasticsearch

**Key Features**:
- Generic MySQL schema extraction and conversion
- Support for both Neo4j (graph) and Elasticsearch (search) targets
- Vector embeddings for similarity search using sentence transformers
- Automatic relationship detection (formal FK + logical relationships)
- Incremental updates and validation
- Command-line interface with extensive options

**Technology Stack**: MySQL Connector, Neo4j Driver, Elasticsearch SDK, SentenceTransformers

### 4. Database Infrastructure (`databases/`)
**Purpose**: Containerized database stack with health monitoring

**Components**:
- **MySQL 8.4**: Source database with sample data initialization
- **Neo4j**: Graph database with APOC plugins for advanced analytics
- **Elasticsearch 8.11**: Search and analytics engine with vector support

**Technology Stack**: Docker Compose, health checks, volume persistence

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Docker and Docker Compose
- OpenAI API key (optional)
- Cohere API key (optional)

### 1. Setup Database Infrastructure
```bash
cd databases/
cp env.example .env
# Edit .env with your database passwords
docker-compose up -d
docker-compose ps  # Verify services are running
```

### 2. Configure API Keys
```bash
# Set environment variables
export OPENAI_API_KEY="your-openai-key"
export COHERE_API_KEY="your-cohere-key"
```

### 3. Launch Chat Interface
```bash
cd chat/
pip install -r requirements.txt
streamlit run main.py
# Access at http://localhost:8501
```

### 4. Start MCP Aggregator (Optional - for tool integration)
```bash
cd mcp-aggregator/
pip install -r requirements.txt
# Edit servers_config.json to configure MCP servers
python main.py
# Server starts on http://localhost:8000/mcp
```

## 🔧 Detailed Component Usage

### Chat Interface Features

- **Multi-Provider LLM Support**: Switch between OpenAI GPT and Cohere Command models
- **Real-time Streaming**: Streaming responses with live token usage tracking
- **MCP Tool Integration**: Connect to MCP servers for dynamic tool calling
- **Cost Monitoring**: Track API usage and estimated costs per conversation
- **Interaction Logging**: Dedicated log viewer for MCP tool executions

**Usage**:
1. Enter API key in sidebar and select your preferred LLM provider
2. For tool integration, enable "MCP Tools" and add server connections
3. Monitor usage and logs in the dedicated tabs

### MCP Aggregator Configuration

The aggregator connects multiple MCP servers into a single HTTP endpoint. Configure servers in `mcp-aggregator/servers_config.json`:

```json
{
  "mcpServers": {
    "neo4j": {
      "command": "uvx",
      "args": ["mcp-neo4j-cypher@0.3.0", "--transport", "stdio"],
      "env": {
        "NEO4J_URI": "bolt://host.docker.internal:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "password"
      }
    }
  }
}
```

The aggregator provides:
- Health monitoring and auto-restart of backend servers
- Dynamic tool registration from all connected servers
- HTTP transport layer for easy client integration

### Database Migration Tool

Advanced MySQL migration supporting both Neo4j (graph) and Elasticsearch (search) targets with vector embeddings.

**Key Features**:
- Generic schema extraction from any MySQL database
- Vector embeddings using SentenceTransformers for semantic search
- Automatic relationship detection (foreign keys + logical patterns)
- Reset and update modes for flexible data management

**Basic Usage**:
```bash
cd mysql-migration/
pip install -r requirements.txt

# Migrate to Neo4j
python main.py --mysql-database employees --target neo4j --mode reset

# Migrate to Elasticsearch with vectors
python main.py --target elasticsearch --enable-vectors --limit 10000

# Semantic search
python main.py --target elasticsearch --search "senior software engineer"
```

## 🔄 Component Interactions

### 1. Chat → MCP Aggregator → MCP Servers
```
User Message → Streamlit → LLM Provider → Tool Calls → 
MCP Aggregator → Specific MCP Server → Tool Execution → 
Results → LLM → Final Response → User
```

### 2. Migration Tool → Databases
```
MySQL Schema → Extraction → Conversion → 
Target DB (Neo4j/Elasticsearch) → Relationship Creation → 
Vector Indexing → Validation
```

### 3. Database Infrastructure
```
Docker Compose → Service Orchestration → 
Health Checks → Persistent Storage → 
Network Communication → External Access
```

## 📊 Data Flow Examples

### MCP Tool Execution Flow
1. User asks: "What files are in my project directory?"
2. Chat interface sends message to selected LLM
3. LLM recognizes need for file system tool
4. Chat interface calls MCP aggregator with tool request
5. MCP aggregator routes to filesystem MCP server
6. Filesystem server executes `list_files` tool
7. Results flow back through the chain
8. LLM incorporates results into natural language response
9. All interactions logged to `mcp_interactions.log`

### Database Migration Flow
1. Migration tool connects to source MySQL database
2. Extracts schema information and foreign key relationships
3. Reads data table by table with optional row limits
4. Converts MySQL data types to target database format
5. Generates vector embeddings for text content (if enabled)
6. Creates nodes/documents in target database
7. Establishes relationships based on foreign keys
8. Detects and creates logical relationships (e.g., matching column names)
9. Creates indexes for performance optimization
10. Validates migration by comparing record counts

## 🛠️ Configuration Files

### Environment Variables
- `OPENAI_API_KEY`: OpenAI API access
- `COHERE_API_KEY`: Cohere API access  
- `NEO4J_PASSWORD`: Neo4j database password
- `MYSQL_ROOT_PASSWORD`: MySQL root password
- `MYSQL_DEVELOPER_PASSWORD`: MySQL developer user password

### Configuration Files
- `mcp-aggregator/servers_config.json`: MCP server definitions
- `databases/.env`: Database credentials
- `databases/docker-compose.yml`: Service definitions
- `databases/init-mysql-db-scripts/`: Database initialization

## 🔍 Advanced Features

### Vector Search and Embeddings
- Uses SentenceTransformers 'all-MiniLM-L6-v2' model
- 384-dimensional embeddings for semantic similarity
- Cosine similarity search across migrated data
- Supports both Neo4j vector indexes and Elasticsearch dense vectors

### Relationship Detection
- **Formal Foreign Keys**: Extracted from MySQL schema
- **Logical Relationships**: Detected by matching column names with primary keys
- **Pattern-Based**: Special handling for common patterns (emp_no → employees, dept_no → departments)

### Health Monitoring
- Docker health checks for all services
- MCP server health monitoring with auto-restart
- Connection validation and error handling
- Comprehensive logging throughout the system

## 🚦 Service Health Monitoring

```bash
# Check database services
cd databases/ && docker-compose ps

# Test MCP aggregator
curl http://localhost:8000/mcp/tools

# Chat interface health
curl http://localhost:8501/_stcore/health
```

All services include health checks and comprehensive logging for easy troubleshooting.

## 🤝 Contributing

- Each component has independent dependencies and can be developed separately
- Follow existing code patterns and maintain comprehensive logging
- Test with sample data before production deployment
- Update documentation when adding new features

## 📝 License

MIT License - provided for educational and development purposes.
