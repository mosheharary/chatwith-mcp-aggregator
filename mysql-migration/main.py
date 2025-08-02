#!/usr/bin/env python3
"""
Enhanced MySQL to Multi-Database Migration Tool
Supports migration to both Neo4j and Elasticsearch with all advanced features:
- Reset/Update modes for target databases
- Generic support for any MySQL schema
- Vector embeddings for similarity search
- Command-line arguments
"""

import mysql.connector
from neo4j import GraphDatabase
from elasticsearch import Elasticsearch
import logging
import argparse
import json
import hashlib
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from sentence_transformers import SentenceTransformer
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MySQLToMultiDBMigrator:
    def __init__(self, mysql_config: Dict, target_config: Dict, target_type: str = "neo4j",
                 mode: str = "reset", enable_vectors: bool = False, 
                 limit: Optional[int] = None):
        self.mysql_config = mysql_config
        self.target_config = target_config
        self.target_type = target_type.lower()  # "neo4j" or "elasticsearch"
        self.mysql_conn = None
        self.target_client = None
        self.schema_info = {}
        self.foreign_keys = {}
        self.mode = mode  # "reset" or "update"
        self.enable_vectors = enable_vectors
        self.limit = limit
        self.model = None
        
        # Initialize sentence transformer for embeddings
        if self.enable_vectors:
            logger.info("Loading sentence transformer model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer model loaded")
        
    def connect_databases(self):
        """Connect to MySQL and target database"""
        try:
            # Connect to MySQL
            self.mysql_conn = mysql.connector.connect(**self.mysql_config)
            logger.info(f"Connected to MySQL database: {self.mysql_config['database']}")
            
            # Connect to target database
            if self.target_type == "neo4j":
                self.target_client = GraphDatabase.driver(
                    self.target_config['uri'],
                    auth=(self.target_config['user'], self.target_config['password'])
                )
                logger.info("Connected to Neo4j database")
                # Test Neo4j connection
                with self.target_client.session() as session:
                    session.run("RETURN 1")
                    
            elif self.target_type == "elasticsearch":
                # Parse host and construct proper URL
                host = self.target_config['host']
                if not host.startswith(('http://', 'https://')):
                    host = f"http://{host}"
                self.target_client = Elasticsearch([host])
                logger.info("Connected to Elasticsearch")
                # Test Elasticsearch connection
                info = self.target_client.info()
                logger.info(f"Elasticsearch cluster: {info['cluster_name']}")
                
        except Exception as e:
            logger.error(f"Failed to connect to databases: {e}")
            raise
    
    def get_mysql_schema(self, database: str = None) -> Dict:
        """Extract MySQL schema information for any database"""
        if database:
            self.mysql_conn.database = database
            
        cursor = self.mysql_conn.cursor(dictionary=True)
        current_db = self.mysql_config.get('database', database)
        
        # Get all tables
        cursor.execute("SHOW TABLES")
        table_results = cursor.fetchall()
        
        # Handle different column names based on MySQL version
        if table_results:
            table_column = list(table_results[0].keys())[0]
            tables = [row[table_column] for row in table_results]
        else:
            tables = []
        
        schema = {}
        for table in tables:
            try:
                # Get column information
                cursor.execute(f"DESCRIBE {table}")
                columns = cursor.fetchall()
                
                # Get primary keys
                cursor.execute(f"""
                    SELECT COLUMN_NAME 
                    FROM information_schema.KEY_COLUMN_USAGE 
                    WHERE TABLE_SCHEMA = '{current_db}' 
                    AND TABLE_NAME = '{table}' 
                    AND CONSTRAINT_NAME = 'PRIMARY'
                """)
                primary_keys = [row['COLUMN_NAME'] for row in cursor.fetchall()]
                
                # Get unique keys
                cursor.execute(f"""
                    SELECT COLUMN_NAME 
                    FROM information_schema.KEY_COLUMN_USAGE 
                    WHERE TABLE_SCHEMA = '{current_db}' 
                    AND TABLE_NAME = '{table}' 
                    AND CONSTRAINT_NAME != 'PRIMARY'
                """)
                unique_keys = [row['COLUMN_NAME'] for row in cursor.fetchall()]
                
                schema[table] = {
                    'columns': columns,
                    'primary_keys': primary_keys,
                    'unique_keys': unique_keys
                }
                
            except Exception as e:
                logger.warning(f"Failed to get schema for table {table}: {e}")
                continue
                
        cursor.close()
        return schema
    
    def get_foreign_keys(self, database: str = None) -> Dict:
        """Extract foreign key relationships for any database"""
        cursor = self.mysql_conn.cursor(dictionary=True)
        current_db = self.mysql_config.get('database', database)
        
        cursor.execute(f"""
            SELECT TABLE_NAME, COLUMN_NAME, CONSTRAINT_NAME, 
                   REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
            FROM information_schema.KEY_COLUMN_USAGE 
            WHERE TABLE_SCHEMA = '{current_db}' 
            AND REFERENCED_TABLE_NAME IS NOT NULL
        """)
        
        foreign_keys = {}
        for row in cursor.fetchall():
            table = row['TABLE_NAME']
            if table not in foreign_keys:
                foreign_keys[table] = []
            foreign_keys[table].append({
                'column': row['COLUMN_NAME'],
                'referenced_table': row['REFERENCED_TABLE_NAME'],
                'referenced_column': row['REFERENCED_COLUMN_NAME']
            })
        
        cursor.close()
        return foreign_keys
    
    def clear_target_database(self):
        """Clear existing target database data"""
        if self.mode == "reset":
            if self.target_type == "neo4j":
                self._clear_neo4j_database()
            elif self.target_type == "elasticsearch":
                self._clear_elasticsearch_database()
        else:
            logger.info("Update mode: keeping existing data")
    
    def _clear_neo4j_database(self):
        """Clear existing Neo4j data"""
        with self.target_client.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            # Try to drop all indexes (Neo4j version dependent)
            try:
                session.run("SHOW INDEXES YIELD name WHERE name <> 'constraint_index' CALL {WITH name DROP INDEX $name} IN TRANSACTIONS")
            except Exception:
                try:
                    session.run("CALL db.indexes() YIELD name CALL db.index.drop(name) RETURN name")
                except Exception:
                    logger.warning("Could not drop indexes - they will be recreated if needed")
            logger.info("Cleared Neo4j database and indexes")
    
    def _clear_elasticsearch_database(self):
        """Clear existing Elasticsearch data"""
        try:
            # Delete all indices that match our pattern
            indices = self.target_client.indices.get_alias(index="*")
            db_prefix = self.mysql_config['database']
            
            for index_name in indices.keys():
                if index_name.startswith(f"{db_prefix}_"):
                    self.target_client.indices.delete(index=index_name)
                    logger.info(f"Deleted Elasticsearch index: {index_name}")
            
            logger.info("Cleared Elasticsearch database")
        except Exception as e:
            logger.warning(f"Could not clear Elasticsearch indices: {e}")
    
    def generate_document_id(self, table_name: str, properties: Dict) -> str:
        """Generate a unique ID for a document based on primary key(s)"""
        schema = self.schema_info[table_name]
        pk_values = []
        
        for pk in schema['primary_keys']:
            pk_values.append(str(properties.get(pk, '')))
        
        # If no primary keys, use all properties sorted by key name
        if not pk_values:
            sorted_items = sorted(properties.items(), key=lambda x: str(x[0]))
            pk_values = [f"{k}:{v}" for k, v in sorted_items]
        
        doc_id = f"{table_name}:{':'.join(pk_values)}"
        return hashlib.md5(doc_id.encode()).hexdigest()
    
    def create_vector_embedding(self, properties: Dict) -> List[float]:
        """Create vector embedding from document properties"""
        if not self.enable_vectors or not self.model:
            return None
        
        # Create text representation of all properties
        text_parts = []
        for key, value in properties.items():
            if value is not None:
                text_parts.append(f"{key}: {value}")
        
        text = " | ".join(text_parts)
        
        # Generate embedding
        embedding = self.model.encode(text)
        return embedding.tolist()
    
    def convert_mysql_value(self, value: Any, column_type: str) -> Any:
        """Convert MySQL value to target database compatible format"""
        if value is None:
            return None
        
        # Handle date/datetime types
        if isinstance(value, (datetime, )):
            return value.isoformat()
        
        # Handle enum types
        if 'enum' in column_type.lower():
            return str(value)
        
        # Handle numeric types
        if column_type.lower().startswith(('int', 'bigint', 'smallint', 'tinyint')):
            return int(value)
        elif column_type.lower().startswith(('decimal', 'float', 'double')):
            return float(value)
        elif column_type.lower().startswith(('varchar', 'char', 'text')):
            return str(value)
        elif column_type.lower().startswith('date'):
            return str(value)
        elif column_type.lower().startswith('json'):
            return json.dumps(value) if isinstance(value, (dict, list)) else str(value)
        
        return str(value)
    
    def migrate_table(self, table_name: str, schema: Dict):
        """Migrate a MySQL table to target database"""
        cursor = self.mysql_conn.cursor(dictionary=True)
        
        # Build query with optional limit
        query = f"SELECT * FROM {table_name}"
        if self.limit:
            query += f" LIMIT {self.limit}"
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        logger.info(f"Processing {len(rows)} rows from {table_name} table")
        
        if self.target_type == "neo4j":
            self._migrate_table_to_neo4j(table_name, schema, rows)
        elif self.target_type == "elasticsearch":
            self._migrate_table_to_elasticsearch(table_name, schema, rows)
        
        cursor.close()
    
    def _migrate_table_to_neo4j(self, table_name: str, schema: Dict, rows: List[Dict]):
        """Migrate table to Neo4j nodes"""
        created_count = 0
        updated_count = 0
        skipped_count = 0
        
        with self.target_client.session() as session:
            for row in rows:
                # Convert MySQL values to Neo4j compatible format
                properties = {}
                for col_name, value in row.items():
                    col_info = next((col for col in schema['columns'] if col['Field'] == col_name), None)
                    if col_info:
                        properties[col_name] = self.convert_mysql_value(value, col_info['Type'])
                
                # Generate node hash
                node_hash = self.generate_document_id(table_name, properties)
                properties['_hash'] = node_hash
                
                # Add vector embedding if enabled
                if self.enable_vectors:
                    embedding = self.create_vector_embedding(properties)
                    if embedding:
                        properties['_embedding'] = embedding
                
                # Check if node exists (only in update mode)
                if self.mode == "update" and self._neo4j_node_exists(table_name, node_hash):
                    # Update existing node
                    session.run(
                        f"MATCH (n:{table_name.capitalize()}) WHERE n._hash = $hash SET n = $props",
                        hash=node_hash,
                        props=properties
                    )
                    updated_count += 1
                elif self.mode == "reset" or not self._neo4j_node_exists(table_name, node_hash):
                    # Create new node
                    session.run(
                        f"CREATE (n:{table_name.capitalize()} $props)",
                        props=properties
                    )
                    created_count += 1
                else:
                    skipped_count += 1
        
        logger.info(f"Neo4j {table_name}: Created={created_count}, Updated={updated_count}, Skipped={skipped_count}")
    
    def _migrate_table_to_elasticsearch(self, table_name: str, schema: Dict, rows: List[Dict]):
        """Migrate table to Elasticsearch documents"""
        created_count = 0
        updated_count = 0
        
        # Create index name
        index_name = f"{self.mysql_config['database']}_{table_name}".lower()
        
        # Create index mapping with vector field if enabled
        if not self.target_client.indices.exists(index=index_name):
            mapping = self._create_elasticsearch_mapping(schema)
            self.target_client.indices.create(index=index_name, body=mapping)
            logger.info(f"Created Elasticsearch index: {index_name}")
        
        for row in rows:
            # Convert MySQL values to Elasticsearch compatible format
            document = {}
            for col_name, value in row.items():
                col_info = next((col for col in schema['columns'] if col['Field'] == col_name), None)
                if col_info:
                    document[col_name] = self.convert_mysql_value(value, col_info['Type'])
            
            # Generate document ID
            doc_id = self.generate_document_id(table_name, document)
            document['_doc_id'] = doc_id
            
            # Add vector embedding if enabled
            if self.enable_vectors:
                embedding = self.create_vector_embedding(document)
                if embedding:
                    document['_embedding'] = embedding
            
            # Add relationship information
            document['_table'] = table_name
            document['_database'] = self.mysql_config['database']
            
            # Check if document exists (only in update mode)
            if self.mode == "update":
                try:
                    existing = self.target_client.get(index=index_name, id=doc_id)
                    # Update existing document
                    self.target_client.index(index=index_name, id=doc_id, body=document)
                    updated_count += 1
                except:
                    # Document doesn't exist, create new
                    self.target_client.index(index=index_name, id=doc_id, body=document)
                    created_count += 1
            else:
                # Reset mode - create/overwrite
                self.target_client.index(index=index_name, id=doc_id, body=document)
                created_count += 1
        
        # Refresh index
        self.target_client.indices.refresh(index=index_name)
        
        logger.info(f"Elasticsearch {table_name}: Created={created_count}, Updated={updated_count}")
    
    def _create_elasticsearch_mapping(self, schema: Dict) -> Dict:
        """Create Elasticsearch mapping based on MySQL schema"""
        mapping = {
            "mappings": {
                "properties": {
                    "_doc_id": {"type": "keyword"},
                    "_table": {"type": "keyword"},
                    "_database": {"type": "keyword"}
                }
            }
        }
        
        # Add vector mapping if enabled
        if self.enable_vectors:
            mapping["mappings"]["properties"]["_embedding"] = {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine"
            }
        
        # Add field mappings based on MySQL columns
        for col in schema['columns']:
            field_name = col['Field']
            col_type = col['Type'].lower()
            
            if col_type.startswith(('int', 'bigint', 'smallint', 'tinyint')):
                mapping["mappings"]["properties"][field_name] = {"type": "long"}
            elif col_type.startswith(('decimal', 'float', 'double')):
                mapping["mappings"]["properties"][field_name] = {"type": "double"}
            elif col_type.startswith('date'):
                mapping["mappings"]["properties"][field_name] = {"type": "date"}
            elif col_type.startswith('bool'):
                mapping["mappings"]["properties"][field_name] = {"type": "boolean"}
            elif col_type.startswith('json'):
                mapping["mappings"]["properties"][field_name] = {"type": "object"}
            else:
                # Default to text with keyword for exact matching
                mapping["mappings"]["properties"][field_name] = {
                    "type": "text",
                    "fields": {
                        "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                        }
                    }
                }
        
        return mapping
    
    def _neo4j_node_exists(self, table_name: str, node_hash: str) -> bool:
        """Check if a node already exists in Neo4j"""
        with self.target_client.session() as session:
            result = session.run(
                f"MATCH (n:{table_name.capitalize()}) WHERE n._hash = $hash RETURN count(n) > 0 as exists",
                hash=node_hash
            )
            return result.single()['exists']
    
    def create_relationships_and_indexes(self):
        """Create relationships and indexes in target database"""
        if self.target_type == "neo4j":
            self._create_neo4j_relationships()
            # Add additional cycle to find missing logical relationships
            self._detect_and_create_missing_relationships()
            self._create_neo4j_indexes()
        elif self.target_type == "elasticsearch":
            self._create_elasticsearch_relationships()
            self._detect_and_create_missing_elasticsearch_relationships()
    
    def _create_neo4j_relationships(self):
        """Create relationships based on foreign key constraints in Neo4j"""
        with self.target_client.session() as session:
            for table_name, fk_list in self.foreign_keys.items():
                for fk in fk_list:
                    # Create relationship between tables
                    rel_name = f"BELONGS_TO_{fk['referenced_table'].upper()}"
                    
                    # In update mode, check if relationships already exist
                    if self.mode == "update":
                        check_query = f"""
                        MATCH (a:{table_name.capitalize()})-[r:{rel_name}]->(b:{fk['referenced_table'].capitalize()})
                        RETURN count(r) as existing_count
                        """
                        existing_result = session.run(check_query)
                        existing_count = existing_result.single()['existing_count']
                        
                        if existing_count > 0:
                            logger.info(f"Skipping existing {rel_name} relationships ({existing_count} found)")
                            continue
                    
                    query = f"""
                    MATCH (a:{table_name.capitalize()}), (b:{fk['referenced_table'].capitalize()})
                    WHERE a.{fk['column']} = b.{fk['referenced_column']}
                    CREATE (a)-[:{rel_name}]->(b)
                    """
                    
                    session.run(query)
                    logger.info(f"Created {rel_name} relationships between {table_name} and {fk['referenced_table']}")
    
    def _create_neo4j_indexes(self):
        """Create indexes in Neo4j"""
        with self.target_client.session() as session:
            for table_name, schema in self.schema_info.items():
                label = table_name.capitalize()
                
                # Create index for hash (always useful for updates)
                try:
                    session.run(f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n._hash)")
                    logger.info(f"Created hash index for {table_name}")
                except Exception as e:
                    logger.warning(f"Failed to create hash index for {table_name}: {e}")
                
                # Create indexes for primary keys
                for pk in schema['primary_keys']:
                    try:
                        session.run(f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.{pk})")
                        logger.info(f"Created index for {table_name}.{pk}")
                    except Exception as e:
                        logger.warning(f"Failed to create index for {table_name}.{pk}: {e}")
                
                # Create vector index if vectors are enabled
                if self.enable_vectors:
                    try:
                        session.run(f"""
                            CREATE VECTOR INDEX `{label.lower()}_embedding` IF NOT EXISTS
                            FOR (n:{label}) ON (n._embedding)
                            OPTIONS {{
                                indexConfig: {{
                                    `vector.dimensions`: 384,
                                    `vector.similarity_function`: 'cosine'
                                }}
                            }}
                        """)
                        logger.info(f"Created vector index for {table_name}")
                    except Exception as e:
                        logger.warning(f"Failed to create vector index for {table_name}: {e}")
    
    def _detect_and_create_missing_relationships(self):
        """Detect and create missing logical relationships in Neo4j"""
        logger.info("Detecting missing logical relationships...")
        
        with self.target_client.session() as session:
            # Common patterns to look for logical relationships
            # Pattern 1: emp_no columns that should reference employees table
            self._create_logical_employee_relationships(session)
            
            # Pattern 2: dept_no columns that should reference departments table  
            self._create_logical_department_relationships(session)
            
            # Pattern 3: Look for any column with same name as primary key in another table
            self._create_generic_logical_relationships(session)
    
    def _create_logical_employee_relationships(self, session):
        """Create relationships for tables with emp_no that reference employees"""
        logger.info("Creating logical employee relationships...")
        
        # Find all tables with emp_no column
        emp_tables = []
        for table_name, schema in self.schema_info.items():
            for col in schema['columns']:
                if col['Field'] == 'emp_no' and table_name != 'employees':
                    emp_tables.append(table_name)
                    break
        
        logger.info(f"Found tables with emp_no: {emp_tables}")
        
        for table_name in emp_tables:
            # Check if this relationship already exists via formal FK
            already_exists = False
            if table_name in self.foreign_keys:
                for fk in self.foreign_keys[table_name]:
                    if fk['referenced_table'] == 'employees' and fk['column'] == 'emp_no':
                        already_exists = True
                        break
            
            if not already_exists:
                rel_name = "BELONGS_TO_EMPLOYEES"
                
                # Check if relationship already exists
                check_query = f"""
                MATCH (a:{table_name.capitalize()})-[r:{rel_name}]->(b:Employees)
                RETURN count(r) as existing_count
                """
                existing_result = session.run(check_query)
                existing_count = existing_result.single()['existing_count']
                
                if existing_count == 0:
                    # Create the relationship
                    query = f"""
                    MATCH (a:{table_name.capitalize()}), (b:Employees)
                    WHERE a.emp_no = b.emp_no
                    CREATE (a)-[:{rel_name}]->(b)
                    """
                    
                    result = session.run(query)
                    summary = result.consume()
                    relationships_created = summary.counters.relationships_created
                    logger.info(f"Created {relationships_created} {rel_name} relationships between {table_name} and employees")
                else:
                    logger.info(f"Skipping {rel_name} for {table_name} - {existing_count} already exist")
    
    def _create_logical_department_relationships(self, session):
        """Create relationships for tables with dept_no that reference departments"""
        logger.info("Creating logical department relationships...")
        
        # Find all tables with dept_no column
        dept_tables = []
        for table_name, schema in self.schema_info.items():
            for col in schema['columns']:
                if col['Field'] == 'dept_no' and table_name != 'departments':
                    dept_tables.append(table_name)
                    break
        
        logger.info(f"Found tables with dept_no: {dept_tables}")
        
        for table_name in dept_tables:
            # Check if this relationship already exists via formal FK
            already_exists = False
            if table_name in self.foreign_keys:
                for fk in self.foreign_keys[table_name]:
                    if fk['referenced_table'] == 'departments' and fk['column'] == 'dept_no':
                        already_exists = True
                        break
            
            if not already_exists:
                rel_name = "BELONGS_TO_DEPARTMENTS"
                
                # Check if relationship already exists
                check_query = f"""
                MATCH (a:{table_name.capitalize()})-[r:{rel_name}]->(b:Departments)
                RETURN count(r) as existing_count
                """
                existing_result = session.run(check_query)
                existing_count = existing_result.single()['existing_count']
                
                if existing_count == 0:
                    # Create the relationship
                    query = f"""
                    MATCH (a:{table_name.capitalize()}), (b:Departments)
                    WHERE a.dept_no = b.dept_no
                    CREATE (a)-[:{rel_name}]->(b)
                    """
                    
                    result = session.run(query)
                    summary = result.consume()
                    relationships_created = summary.counters.relationships_created
                    logger.info(f"Created {relationships_created} {rel_name} relationships between {table_name} and departments")
                else:
                    logger.info(f"Skipping {rel_name} for {table_name} - {existing_count} already exist")
    
    def _create_generic_logical_relationships(self, session):
        """Create relationships based on matching column names and primary keys"""
        logger.info("Creating generic logical relationships...")
        
        # Get all primary key columns
        pk_columns = {}
        for table_name, schema in self.schema_info.items():
            for pk in schema['primary_keys']:
                if pk not in pk_columns:
                    pk_columns[pk] = []
                pk_columns[pk].append(table_name)
        
        logger.info(f"Primary key columns found: {pk_columns}")
        
        # Look for foreign key patterns
        for table_name, schema in self.schema_info.items():
            for col in schema['columns']:
                col_name = col['Field']
                
                # If this column name matches a primary key in another table
                if col_name in pk_columns:
                    for target_table in pk_columns[col_name]:
                        if target_table != table_name:  # Don't create self-references
                            # Check if formal FK already exists
                            already_exists = False
                            if table_name in self.foreign_keys:
                                for fk in self.foreign_keys[table_name]:
                                    if (fk['referenced_table'] == target_table and 
                                        fk['column'] == col_name):
                                        already_exists = True
                                        break
                            
                            if not already_exists:
                                rel_name = f"BELONGS_TO_{target_table.upper()}"
                                
                                # Check if relationship already exists
                                check_query = f"""
                                MATCH (a:{table_name.capitalize()})-[r:{rel_name}]->(b:{target_table.capitalize()})
                                RETURN count(r) as existing_count
                                """
                                existing_result = session.run(check_query)
                                existing_count = existing_result.single()['existing_count']
                                
                                if existing_count == 0:
                                    # Check if there are actually matching values
                                    test_query = f"""
                                    MATCH (a:{table_name.capitalize()}), (b:{target_table.capitalize()})
                                    WHERE a.{col_name} = b.{col_name}
                                    RETURN count(*) as potential_matches
                                    """
                                    test_result = session.run(test_query)
                                    potential_matches = test_result.single()['potential_matches']
                                    
                                    if potential_matches > 0:
                                        # Create the relationship
                                        query = f"""
                                        MATCH (a:{table_name.capitalize()}), (b:{target_table.capitalize()})
                                        WHERE a.{col_name} = b.{col_name}
                                        CREATE (a)-[:{rel_name}]->(b)
                                        """
                                        
                                        result = session.run(query)
                                        summary = result.consume()
                                        relationships_created = summary.counters.relationships_created
                                        logger.info(f"Created {relationships_created} {rel_name} relationships between {table_name} and {target_table}")
                                    else:
                                        logger.info(f"No matching values for {table_name}.{col_name} -> {target_table}.{col_name}")
                                else:
                                    logger.info(f"Skipping {rel_name} for {table_name} -> {target_table} - {existing_count} already exist")
    
    def _detect_and_create_missing_elasticsearch_relationships(self):
        """Detect and create missing logical relationships in Elasticsearch"""
        logger.info("Detecting missing logical relationships for Elasticsearch...")
        
        # For Elasticsearch, we'll add relationship metadata to documents
        # that don't have explicit foreign keys but have logical relationships
        
        # Find logical relationships similar to Neo4j
        logical_relationships = self._find_logical_relationships()
        
        for source_table, relationships in logical_relationships.items():
            index_name = f"{self.mysql_config['database']}_{source_table}".lower()
            
            try:
                # Add logical relationship metadata
                query = {
                    "script": {
                        "source": """
                        if (ctx._source._logical_relationships == null) {
                            ctx._source._logical_relationships = params.logical_relationships;
                        } else {
                            ctx._source._logical_relationships.addAll(params.logical_relationships);
                        }
                        """,
                        "params": {
                            "logical_relationships": [
                                {
                                    "type": f"logical_belongs_to_{rel['target_table']}",
                                    "foreign_key": rel['source_column'],
                                    "referenced_table": rel['target_table'],
                                    "referenced_key": rel['target_column']
                                }
                                for rel in relationships
                            ]
                        }
                    },
                    "query": {"match_all": {}}
                }
                
                self.target_client.update_by_query(
                    index=index_name, 
                    body=query,
                    conflicts='proceed',  # Continue despite version conflicts
                    refresh=True         # Refresh index after operation
                )
                logger.info(f"Added {len(relationships)} logical relationship metadata to {source_table} documents")
                
            except Exception as e:
                logger.warning(f"Failed to add logical relationships to {source_table}: {e}")
    
    def _find_logical_relationships(self):
        """Find logical relationships that aren't formally defined as foreign keys"""
        logical_relationships = {}
        
        # Get all primary key columns
        pk_columns = {}
        for table_name, schema in self.schema_info.items():
            for pk in schema['primary_keys']:
                if pk not in pk_columns:
                    pk_columns[pk] = []
                pk_columns[pk].append(table_name)
        
        # Look for foreign key patterns
        for table_name, schema in self.schema_info.items():
            for col in schema['columns']:
                col_name = col['Field']
                
                # If this column name matches a primary key in another table
                if col_name in pk_columns:
                    for target_table in pk_columns[col_name]:
                        if target_table != table_name:  # Don't create self-references
                            # Check if formal FK already exists
                            already_exists = False
                            if table_name in self.foreign_keys:
                                for fk in self.foreign_keys[table_name]:
                                    if (fk['referenced_table'] == target_table and 
                                        fk['column'] == col_name):
                                        already_exists = True
                                        break
                            
                            if not already_exists:
                                if table_name not in logical_relationships:
                                    logical_relationships[table_name] = []
                                
                                logical_relationships[table_name].append({
                                    'source_column': col_name,
                                    'target_table': target_table,
                                    'target_column': col_name
                                })
        
        return logical_relationships
    
    def _create_elasticsearch_relationships(self):
        """Store relationship information in Elasticsearch documents"""
        # Elasticsearch doesn't have explicit relationships like Neo4j
        # But we can add relationship fields to documents for joins
        
        for table_name, fk_list in self.foreign_keys.items():
            index_name = f"{self.mysql_config['database']}_{table_name}".lower()
            
            try:
                # Update documents with relationship information
                query = {
                    "script": {
                        "source": "ctx._source._relationships = params.relationships",
                        "params": {
                            "relationships": [
                                {
                                    "type": f"belongs_to_{fk['referenced_table']}",
                                    "foreign_key": fk['column'],
                                    "referenced_table": fk['referenced_table'],
                                    "referenced_key": fk['referenced_column']
                                }
                                for fk in fk_list
                            ]
                        }
                    },
                    "query": {"match_all": {}}
                }
                
                self.target_client.update_by_query(
                    index=index_name, 
                    body=query,
                    conflicts='proceed',  # Continue despite version conflicts
                    refresh=True         # Refresh index after operation
                )
                logger.info(f"Added relationship metadata to {table_name} documents")
                
            except Exception as e:
                logger.warning(f"Failed to add relationships to {table_name}: {e}")
    
    def migrate(self, database: str = None):
        """Main migration method"""
        try:
            # Connect to databases
            self.connect_databases()
            
            # Use specified database or default
            target_db = database or self.mysql_config['database']
            
            # Get schema information
            logger.info(f"Extracting MySQL schema for database: {target_db}")
            self.schema_info = self.get_mysql_schema(target_db)
            self.foreign_keys = self.get_foreign_keys(target_db)
            
            logger.info(f"Found {len(self.schema_info)} tables: {list(self.schema_info.keys())}")
            logger.info(f"Found foreign keys in {len(self.foreign_keys)} tables")
            
            # Clear target database if in reset mode
            self.clear_target_database()
            
            # Migrate each table
            logger.info(f"Migrating tables to {self.target_type} (mode: {self.mode})...")
            for table_name, schema in self.schema_info.items():
                self.migrate_table(table_name, schema)
            
            # Create relationships and indexes
            logger.info("Creating relationships and indexes...")
            self.create_relationships_and_indexes()
            
            logger.info(f"Migration to {self.target_type} completed successfully!")
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise
        finally:
            # Close connections
            if self.mysql_conn:
                self.mysql_conn.close()
            if self.target_client and self.target_type == "neo4j":
                self.target_client.close()
    
    def similarity_search(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """Perform similarity search using vector embeddings"""
        if not self.enable_vectors or not self.model:
            logger.error("Vector embeddings not enabled")
            return []
        
        try:
            self.connect_databases()
            
            # Generate embedding for query
            query_embedding = self.model.encode(query_text).tolist()
            
            results = []
            
            if self.target_type == "neo4j":
                results = self._neo4j_similarity_search(query_embedding, top_k)
            elif self.target_type == "elasticsearch":
                results = self._elasticsearch_similarity_search(query_embedding, top_k)
            
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
        finally:
            if self.target_client and self.target_type == "neo4j":
                self.target_client.close()
    
    def _neo4j_similarity_search(self, query_embedding: List[float], top_k: int) -> List[Dict]:
        """Perform similarity search in Neo4j"""
        results = []
        with self.target_client.session() as session:
            for table_name in self.schema_info.keys():
                label = table_name.capitalize()
                
                # Perform vector similarity search
                query = f"""
                MATCH (n:{label})
                WHERE n._embedding IS NOT NULL
                CALL db.index.vector.queryNodes('{label.lower()}_embedding', {top_k}, $embedding)
                YIELD node, score
                RETURN node, score, '{table_name}' as table_name
                ORDER BY score DESC
                """
                
                try:
                    result = session.run(query, embedding=query_embedding)
                    for record in result:
                        node_props = dict(record['node'])
                        # Remove internal properties
                        node_props.pop('_hash', None)
                        node_props.pop('_embedding', None)
                        
                        results.append({
                            'table': record['table_name'],
                            'score': record['score'],
                            'properties': node_props
                        })
                except Exception as e:
                    logger.warning(f"Vector search failed for {table_name}: {e}")
                    continue
        
        # Sort all results by score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def _elasticsearch_similarity_search(self, query_embedding: List[float], top_k: int) -> List[Dict]:
        """Perform similarity search in Elasticsearch"""
        results = []
        
        # Search across all indices
        db_prefix = self.mysql_config['database']
        indices = self.target_client.indices.get_alias(index=f"{db_prefix}_*")
        
        for index_name in indices.keys():
            try:
                search_query = {
                    "size": top_k,
                    "query": {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, '_embedding') + 1.0",
                                "params": {"query_vector": query_embedding}
                            }
                        }
                    },
                    "_source": {"excludes": ["_embedding"]}
                }
                
                response = self.target_client.search(index=index_name, body=search_query)
                
                for hit in response['hits']['hits']:
                    table_name = hit['_source'].get('_table', index_name.split('_')[-1])
                    source = hit['_source'].copy()
                    # Remove internal properties
                    source.pop('_doc_id', None)
                    source.pop('_table', None)
                    source.pop('_database', None)
                    source.pop('_relationships', None)
                    
                    results.append({
                        'table': table_name,
                        'score': hit['_score'],
                        'properties': source
                    })
                    
            except Exception as e:
                logger.warning(f"Vector search failed for index {index_name}: {e}")
                continue
        
        # Sort all results by score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def validate_migration(self, database: str = None):
        """Validate the migration by comparing counts"""
        try:
            self.connect_databases()
            
            target_db = database or self.mysql_config['database']
            mysql_cursor = self.mysql_conn.cursor()
            
            total_mysql = 0
            total_target = 0
            
            for table_name in self.schema_info.keys():
                # Get MySQL count
                query = f"SELECT COUNT(*) FROM {table_name}"
                if self.limit:
                    query = f"SELECT COUNT(*) FROM (SELECT * FROM {table_name} LIMIT {self.limit}) as limited"
                
                mysql_cursor.execute(query)
                mysql_count = mysql_cursor.fetchone()[0]
                
                # Get target database count
                if self.target_type == "neo4j":
                    with self.target_client.session() as session:
                        result = session.run(f"MATCH (n:{table_name.capitalize()}) RETURN count(n) as count")
                        target_count = result.single()['count']
                elif self.target_type == "elasticsearch":
                    index_name = f"{target_db}_{table_name}".lower()
                    try:
                        response = self.target_client.count(index=index_name)
                        target_count = response['count']
                    except:
                        target_count = 0
                
                logger.info(f"{table_name}: MySQL={mysql_count}, {self.target_type.capitalize()}={target_count}")
                
                total_mysql += mysql_count
                total_target += target_count
                
                if mysql_count != target_count:
                    logger.warning(f"Count mismatch for {table_name}!")
                else:
                    logger.info(f"✓ {table_name} counts match")
            
            logger.info(f"Total: MySQL={total_mysql}, {self.target_type.capitalize()}={total_target}")
            
            mysql_cursor.close()
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise
        finally:
            if self.mysql_conn:
                self.mysql_conn.close()
            if self.target_client and self.target_type == "neo4j":
                self.target_client.close()

def main():
    parser = argparse.ArgumentParser(description='MySQL to Multi-Database Migration Tool')
    
    # MySQL configuration
    parser.add_argument('--mysql-host', default='localhost', help='MySQL host')
    parser.add_argument('--mysql-port', type=int, default=3306, help='MySQL port')
    parser.add_argument('--mysql-user', default='root', help='MySQL username')
    parser.add_argument('--mysql-password', default='rootpassword', help='MySQL password')
    parser.add_argument('--mysql-database', default='employees', help='MySQL database name')
    
    # Target database selection
    parser.add_argument('--target', choices=['neo4j', 'elasticsearch'], default='neo4j',
                        help='Target database type')
    
    # Neo4j configuration
    parser.add_argument('--neo4j-uri', default='bolt://localhost:7687', help='Neo4j URI')
    parser.add_argument('--neo4j-user', default='neo4j', help='Neo4j username')
    parser.add_argument('--neo4j-password', default='password', help='Neo4j password')
    
    # Elasticsearch configuration
    parser.add_argument('--elasticsearch-host', default='localhost:9200', help='Elasticsearch host:port')
    
    # Migration options
    parser.add_argument('--mode', choices=['reset', 'update'], default='reset', 
                        help='Migration mode: reset (clear all) or update (incremental)')
    parser.add_argument('--enable-vectors', action='store_true', 
                        help='Enable vector embeddings for similarity search')
    parser.add_argument('--limit', type=int, help='Limit number of rows to migrate (for testing)')
    parser.add_argument('--validate', action='store_true', help='Validate migration after completion')
    parser.add_argument('--search', type=str, help='Perform similarity search with given query')
    
    args = parser.parse_args()
    
    # MySQL configuration
    mysql_config = {
        'host': args.mysql_host,
        'port': args.mysql_port,
        'user': args.mysql_user,
        'password': args.mysql_password,
        'database': args.mysql_database
    }
    
    # Target database configuration
    if args.target == 'neo4j':
        target_config = {
            'uri': args.neo4j_uri,
            'user': args.neo4j_user,
            'password': args.neo4j_password
        }
    else:  # elasticsearch
        target_config = {
            'host': args.elasticsearch_host
        }
    
    # Create migrator instance
    migrator = MySQLToMultiDBMigrator(
        mysql_config, 
        target_config,
        target_type=args.target,
        mode=args.mode,
        enable_vectors=args.enable_vectors,
        limit=args.limit
    )
    
    try:
        if args.search:
            # Perform similarity search
            logger.info(f"Performing similarity search in {args.target} for: {args.search}")
            results = migrator.similarity_search(args.search)
            
            if results:
                logger.info(f"Found {len(results)} similar results:")
                for i, result in enumerate(results, 1):
                    logger.info(f"{i}. Table: {result['table']}, Score: {result['score']:.4f}")
                    logger.info(f"   Properties: {result['properties']}")
            else:
                logger.info("No results found")
        else:
            # Run migration
            migrator.migrate()
            
            # Validate migration if requested
            if args.validate:
                logger.info("Validating migration...")
                migrator.validate_migration()
                
    except KeyboardInterrupt:
        logger.info("Migration interrupted by user")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise

if __name__ == "__main__":
    main()