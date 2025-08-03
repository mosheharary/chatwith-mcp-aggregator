import streamlit as st
from openai import OpenAI
import cohere
import os
import tiktoken
import json
import asyncio
import logging
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, List, Iterator, Tuple, Optional, Any
from dotenv import load_dotenv
from fastmcp import Client
from enum import Enum
from dataclasses import dataclass, asdict
import subprocess
import stat
import tempfile

# Load environment variables
load_dotenv()

# Asyncio Event Loop Helper
def run_async_in_streamlit(coro):
    """Safely run async functions in Streamlit environment"""
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, we can't use run_until_complete
            # This is common in Jupyter/Streamlit environments
            try:
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(coro)
            except ImportError:
                st.warning("nest_asyncio not available, trying alternative approach")
                # Create a new thread to run the async function
                import concurrent.futures
                import threading
                
                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(coro)
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop exists, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            # Don't close the loop as it might be needed for other operations
            pass
    except Exception as e:
        st.error(f"Event loop error: {str(e)}")
        # Fallback: create a completely new event loop
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            pass

# Transport Protocol Enums and Configuration Classes
class TransportProtocol(Enum):
    STDIO = "stdio"
    HTTP = "http"
    WEBSOCKET = "websocket"
    SERVER_CONFIG = "server_config"

@dataclass
class ServerConfig:
    name: str
    transport: TransportProtocol
    config: Dict[str, Any]
    
    def to_dict(self):
        return {
            "name": self.name,
            "transport": self.transport.value,
            "config": self.config
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(
            name=data["name"],
            transport=TransportProtocol(data["transport"]),
            config=data["config"]
        )

@dataclass
class StdioConfig:
    command: str
    args: List[str] = None
    env: Dict[str, str] = None
    working_dir: str = None
    
    def __post_init__(self):
        if self.args is None:
            self.args = []
        if self.env is None:
            self.env = {}

@dataclass
class HttpConfig:
    url: str
    headers: Dict[str, str] = None
    timeout: int = 30
    verify_ssl: bool = True
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {}

@dataclass
class WebSocketConfig:
    url: str
    headers: Dict[str, str] = None
    timeout: int = 30
    ping_interval: int = 20
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {}

@dataclass
class ServerConfigReference:
    config_name: str
    config_path: str = "servers_config.json"

# Configure MCP logging
def setup_mcp_logging():
    """Setup dedicated logging for MCP interactions"""
    mcp_logger = logging.getLogger('mcp_interactions')
    mcp_logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    for handler in mcp_logger.handlers[:]:
        mcp_logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler('mcp_interactions.log')
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    mcp_logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    mcp_logger.propagate = False
    
    return mcp_logger

# Initialize MCP logger
mcp_logger = setup_mcp_logging()

# Abstract LLM Provider Base Class
class LLMProvider(ABC):
    @abstractmethod
    def get_models(self) -> List[str]:
        pass
    
    @abstractmethod
    def chat_completion(self, messages: List[Dict], model: str, api_key: str, tools: Optional[List[Dict]] = None) -> Iterator[str]:
        pass
    
    @abstractmethod
    def count_tokens(self, messages: List[Dict], model: str) -> int:
        pass
    
    @abstractmethod
    def get_pricing(self) -> Dict[str, Dict[str, float]]:
        pass

# OpenAI Provider Implementation
class OpenAIProvider(LLMProvider):
    def get_models(self) -> List[str]:
        return [
            "gpt-4o",
            "gpt-4o-mini", 
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo"
        ]
    
    def chat_completion(self, messages: List[Dict], model: str, api_key: str, tools: Optional[List[Dict]] = None) -> Iterator[str]:
        # Clear proxy settings
        os.environ.pop('HTTP_PROXY', None)
        os.environ.pop('HTTPS_PROXY', None)
        os.environ.pop('http_proxy', None)
        os.environ.pop('https_proxy', None)
        
        client = OpenAI(api_key=api_key, base_url=None)
        
        # If tools are provided, use non-streaming approach for better tool handling
        if tools and len(tools) > 0:
            # Store the MCP manager from session state for tool execution
            mcp_manager = st.session_state.mcp_manager
            result = handle_tool_calls_with_openai(client, messages.copy(), model, tools, mcp_manager)
            yield result
            return
        
        # Regular streaming for non-tool calls
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    
    def count_tokens(self, messages: List[Dict], model: str) -> int:
        try:
            if model in ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]:
                encoding = tiktoken.encoding_for_model("gpt-4")
            else:
                encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        
        tokens_per_message = 3
        tokens_per_name = 1
        
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        
        num_tokens += 3  # Every reply is primed with assistant
        return num_tokens
    
    def get_pricing(self) -> Dict[str, Dict[str, float]]:
        return {
            "gpt-4o": {"prompt": 0.005, "completion": 0.015},
            "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
            "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
            "gpt-4": {"prompt": 0.03, "completion": 0.06},
            "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015}
        }

# Cohere Provider Implementation
class CohereProvider(LLMProvider):
    def get_models(self) -> List[str]:
        return [
            "command-r-plus",
            "command-r",
            "command"
        ]
    
    def chat_completion(self, messages: List[Dict], model: str, api_key: str, tools: Optional[List[Dict]] = None) -> Iterator[str]:
        client = cohere.ClientV2(api_key=api_key)
        
        # Convert messages to Cohere format
        cohere_messages = []
        for msg in messages:
            cohere_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Use non-streaming first, then implement streaming if available
        try:
            response = client.chat(
                model=model,
                messages=cohere_messages
            )
            # Return the full response as a single chunk for now
            if hasattr(response, 'message') and hasattr(response.message, 'content'):
                for content_item in response.message.content:
                    if hasattr(content_item, 'text'):
                        yield content_item.text
            else:
                yield str(response)
        except Exception as e:
            # Fallback for different API response structure
            yield f"Error: {str(e)}"
    
    def count_tokens(self, messages: List[Dict], model: str) -> int:
        # Cohere doesn't have a direct token counting API like OpenAI
        # Using approximate calculation: ~4 characters per token
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        return int(total_chars / 4)
    
    def get_pricing(self) -> Dict[str, Dict[str, float]]:
        # Cohere pricing (approximate, as of 2024)
        return {
            "command-r-plus": {"prompt": 0.003, "completion": 0.015},
            "command-r": {"prompt": 0.0005, "completion": 0.0015},
            "command": {"prompt": 0.001, "completion": 0.002}
        }

# MCP Manager for handling MCP server connections and tools
class MCPManager:
    def __init__(self):
        self.clients: Dict[str, Client] = {}
        self.tools_cache: Dict[str, List[Dict]] = {}
        self.resources_cache: Dict[str, List[Dict]] = {}
        self.server_configs: Dict[str, ServerConfig] = {}
        self.connection_status: Dict[str, str] = {}
        
    async def add_server_with_config(self, server_config: ServerConfig) -> bool:
        """Add a new MCP server using ServerConfig object"""
        try:
            # Validate inputs
            if not server_config.name:
                st.error("Server name is required")
                return False
            
            # Check if server name already exists
            if server_config.name in self.clients:
                st.error(f"Server with name '{server_config.name}' already exists")
                return False
            
            client = await self._create_client_from_config(server_config)
            if not client:
                return False
            
            # Test connection and cache tools/resources with timeout
            try:
                async with client:
                    tools = await client.list_tools()
                    resources = await client.list_resources()
                
                    # Safely convert to dict
                    self.tools_cache[server_config.name] = []
                    self.resources_cache[server_config.name] = []
                    
                    for tool in tools:
                        try:
                            self.tools_cache[server_config.name].append(tool.model_dump())
                        except Exception:
                            # Fallback for tools that can't be dumped
                            self.tools_cache[server_config.name].append({
                                "name": getattr(tool, 'name', 'unknown'),
                                "description": getattr(tool, 'description', ''),
                                "server": server_config.name
                            })
                    
                    for resource in resources:
                        try:
                            self.resources_cache[server_config.name].append(resource.model_dump())
                        except Exception:
                            # Fallback for resources that can't be dumped
                            self.resources_cache[server_config.name].append({
                                "name": getattr(resource, 'name', 'unknown'),
                                "description": getattr(resource, 'description', ''),
                                "server": server_config.name
                            })
                
                self.clients[server_config.name] = client
                self.server_configs[server_config.name] = server_config
                self.connection_status[server_config.name] = "connected"
                return True
                
            except Exception as connection_error:
                st.error(f"Connection test failed: {str(connection_error)}")
                self.connection_status[server_config.name] = "connection_error"
                return False
            
        except ConnectionError as e:
            self.connection_status[server_config.name] = "connection_error"
            st.error(f"Connection failed for server '{server_config.name}': {str(e)}")
            return False
        except TimeoutError as e:
            self.connection_status[server_config.name] = "timeout"
            st.error(f"Connection timeout for server '{server_config.name}': {str(e)}")
            return False
        except Exception as e:
            self.connection_status[server_config.name] = "error"
            st.error(f"Failed to connect to MCP server '{server_config.name}': {str(e)}")
            return False
    
    async def _create_client_from_config(self, server_config: ServerConfig) -> Optional[Client]:
        """Create a FastMCP Client based on the server configuration"""
        try:
            if server_config.transport == TransportProtocol.HTTP:
                http_config = HttpConfig(**server_config.config)
                return Client(http_config.url)
            
            elif server_config.transport == TransportProtocol.STDIO:
                stdio_config = StdioConfig(**server_config.config)
                
                # Create a Python wrapper script that FastMCP can understand
                full_command = [stdio_config.command] + stdio_config.args
                
                # Build environment variables dict
                env_vars = {}
                if stdio_config.env:
                    env_vars.update(stdio_config.env)
                
                # Create Python wrapper script
                wrapper_content = f"""#!/usr/bin/env python3
import subprocess
import os
import sys

# Set environment variables
env = os.environ.copy()
env.update({repr(env_vars)})

# Change working directory if specified
working_dir = {repr(stdio_config.working_dir)}
if working_dir:
    os.chdir(working_dir)

# Build command
command = {repr(full_command)}

# Execute the command
try:
    result = subprocess.run(command, env=env, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)
    sys.exit(result.returncode)
except Exception as e:
    print(f"Error executing command: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
                
                # Write to temporary Python file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(wrapper_content)
                    wrapper_path = f.name
                
                # Make executable
                os.chmod(wrapper_path, stat.S_IRWXU)
                
                return Client(wrapper_path)
            
            elif server_config.transport == TransportProtocol.WEBSOCKET:
                ws_config = WebSocketConfig(**server_config.config)
                return Client(ws_config.url)
            
            elif server_config.transport == TransportProtocol.SERVER_CONFIG:
                config_ref = ServerConfigReference(**server_config.config)
                # Load from external config file
                return await self._create_client_from_server_config(config_ref)
            
            # Try value-based comparison as fallback
            elif hasattr(server_config.transport, 'value'):
                transport_value = server_config.transport.value
                
                if transport_value == "stdio":
                    stdio_config = StdioConfig(**server_config.config)
                    
                    # Create Python wrapper script (same as above)
                    full_command = [stdio_config.command] + stdio_config.args
                    env_vars = {}
                    if stdio_config.env:
                        env_vars.update(stdio_config.env)
                    
                    wrapper_content = f"""#!/usr/bin/env python3
import subprocess
import os
import sys

# Set environment variables
env = os.environ.copy()
env.update({repr(env_vars)})

# Change working directory if specified
working_dir = {repr(stdio_config.working_dir)}
if working_dir:
    os.chdir(working_dir)

# Build command
command = {repr(full_command)}

# Execute the command
try:
    result = subprocess.run(command, env=env, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)
    sys.exit(result.returncode)
except Exception as e:
    print(f"Error executing command: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
                    
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                        f.write(wrapper_content)
                        wrapper_path = f.name
                    
                    os.chmod(wrapper_path, stat.S_IRWXU)
                    return Client(wrapper_path)
                
                elif transport_value == "http":
                    http_config = HttpConfig(**server_config.config)
                    return Client(http_config.url)
                
                elif transport_value == "websocket":
                    ws_config = WebSocketConfig(**server_config.config)
                    return Client(ws_config.url)
                
                elif transport_value == "server_config":
                    config_ref = ServerConfigReference(**server_config.config)
                    return await self._create_client_from_server_config(config_ref)
                
                else:
                    st.error(f"❌ Unsupported transport value: {transport_value}")
                    return None
            
            else:
                st.error(f"❌ Unsupported transport protocol: {server_config.transport} (no value attribute)")
                st.error(f"❌ Transport type: {type(server_config.transport)}")
                st.error(f"❌ Available protocols: {list(TransportProtocol)}")
                return None
                
        except Exception as e:
            st.error(f"Error creating client: {str(e)}")
            return None
    
    async def _create_client_from_server_config(self, config_ref: ServerConfigReference) -> Optional[Client]:
        """Create client from external server config file"""
        try:
            config_path = config_ref.config_path
            if not os.path.exists(config_path):
                st.error(f"Server config file not found: {config_path}")
                return None
            
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            if 'mcpServers' not in config_data:
                st.error("Invalid server config format: missing 'mcpServers'")
                return None
            
            server_data = config_data['mcpServers'].get(config_ref.config_name)
            if not server_data:
                st.error(f"Server '{config_ref.config_name}' not found in config")
                return None
            
            # Create a temporary script that runs the command
            
            command = server_data.get('command', '')
            args = server_data.get('args', [])
            env_vars = server_data.get('env', {})
            
            # For now, we'll treat server config as stdio transport
            full_command = [command] + args
            
            # Create a temporary wrapper script
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(f"""
import subprocess
import os
import sys

# Set environment variables
env = os.environ.copy()
env.update({repr(env_vars)})

# Run the command
subprocess.run({repr(full_command)}, env=env)
""")
                temp_script = f.name
            
            return Client(temp_script)
            
        except Exception as e:
            st.error(f"Error loading server config: {str(e)}")
            return None
    
    async def test_connection(self, server_config: ServerConfig) -> bool:
        """Test connection to a server without adding it permanently"""
        try:
            client = await self._create_client_from_config(server_config)
            if not client:
                return False
            
            async with client:
                await client.ping()
                return True
                
        except Exception as e:
            st.error(f"Connection test failed: {str(e)}")
            return False
    
    async def add_server(self, name: str, server_url_or_path: str) -> bool:
        """Add a new MCP server connection"""
        try:
            # Validate inputs
            if not name or not server_url_or_path:
                st.error("Server name and URL/path are required")
                return False
            
            # Check if server name already exists
            if name in self.clients:
                st.error(f"Server with name '{name}' already exists")
                return False
            
            client = Client(server_url_or_path)
            
            # Test connection and cache tools/resources with timeout
            async with client:
                tools = await client.list_tools()
                resources = await client.list_resources()
                
                # Safely convert to dict
                self.tools_cache[name] = []
                self.resources_cache[name] = []
                
                for tool in tools:
                    try:
                        self.tools_cache[name].append(tool.model_dump())
                    except Exception:
                        # Fallback for tools that can't be dumped
                        self.tools_cache[name].append({
                            "name": getattr(tool, 'name', 'unknown'),
                            "description": getattr(tool, 'description', ''),
                            "server": name
                        })
                
                for resource in resources:
                    try:
                        self.resources_cache[name].append(resource.model_dump())
                    except Exception:
                        # Fallback for resources that can't be dumped
                        self.resources_cache[name].append({
                            "name": getattr(resource, 'name', 'unknown'),
                            "description": getattr(resource, 'description', ''),
                            "server": name
                        })
            
            self.clients[name] = client
            return True
            
        except ConnectionError as e:
            st.error(f"Connection failed for server '{name}': {str(e)}")
            return False
        except TimeoutError as e:
            st.error(f"Connection timeout for server '{name}': {str(e)}")
            return False
        except Exception as e:
            st.error(f"Failed to connect to MCP server '{name}': {str(e)}")
            return False
    
    def remove_server(self, name: str):
        """Remove an MCP server connection"""
        if name in self.clients:
            del self.clients[name]
        if name in self.tools_cache:
            del self.tools_cache[name]
        if name in self.resources_cache:
            del self.resources_cache[name]
        if name in self.server_configs:
            del self.server_configs[name]
        if name in self.connection_status:
            del self.connection_status[name]
    
    def get_server_config(self, name: str) -> Optional[ServerConfig]:
        """Get server configuration by name"""
        return self.server_configs.get(name)
    
    def get_connection_status(self, name: str) -> str:
        """Get connection status for a server"""
        return self.connection_status.get(name, "unknown")
    
    def export_server_configs(self) -> Dict[str, Any]:
        """Export all server configurations to a dictionary"""
        return {
            name: config.to_dict() 
            for name, config in self.server_configs.items()
        }
    
    def import_server_configs(self, configs_data: Dict[str, Any]):
        """Import server configurations from a dictionary"""
        for name, config_data in configs_data.items():
            try:
                config = ServerConfig.from_dict(config_data)
                self.server_configs[name] = config
            except Exception as e:
                st.warning(f"Failed to import config for '{name}': {str(e)}")
    
    async def load_predefined_servers(self, config_path: str = "../mcp-aggregator/servers_config.json"):
        """Load servers from the aggregator's config file"""
        try:
            if not os.path.exists(config_path):
                return []
            
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            available_servers = []
            if 'mcpServers' in config_data:
                for server_name, server_data in config_data['mcpServers'].items():
                    available_servers.append({
                        'name': server_name,
                        'command': server_data.get('command', ''),
                        'args': server_data.get('args', []),
                        'env': server_data.get('env', {})
                    })
            
            return available_servers
            
        except Exception as e:
            st.warning(f"Failed to load predefined servers: {str(e)}")
            return []
    
    async def execute_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool on a specific MCP server"""
        if server_name not in self.clients:
            raise ValueError(f"Server '{server_name}' not found")
        
        client = self.clients[server_name]
        async with client:
            result = await client.call_tool(tool_name, arguments)
            return {
                "success": True,
                "result": result.data if hasattr(result, 'data') else str(result),
                "content": result.content if hasattr(result, 'content') else []
            }
    
    def get_all_tools(self) -> List[Dict[str, Any]]:
        """Get all tools from all connected servers"""
        all_tools = []
        for server_name, tools in self.tools_cache.items():
            for tool in tools:
                tool_info = tool.copy()
                tool_info['server'] = server_name
                all_tools.append(tool_info)
        return all_tools
    
    def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """Get tools formatted for LLM function calling"""
        all_tools = self.get_all_tools()
        formatted_tools = []
        
        for tool in all_tools:
            formatted_tool = {
                "type": "function",
                "function": {
                    "name": f"{tool['server']}_{tool['name']}",
                    "description": tool.get('description', ''),
                    "parameters": tool.get('inputSchema', {})
                }
            }
            formatted_tools.append(formatted_tool)
        
        return formatted_tools
    
    def get_connected_servers(self) -> List[str]:
        """Get list of connected server names"""
        return list(self.clients.keys())
    
    async def refresh_tools_cache(self):
        """Refresh the tools and resources cache for all servers"""
        for name, client in self.clients.items():
            try:
                async with client:
                    tools = await client.list_tools()
                    resources = await client.list_resources()
                    self.tools_cache[name] = [tool.model_dump() for tool in tools]
                    self.resources_cache[name] = [resource.model_dump() for resource in resources]
            except Exception as e:
                st.warning(f"Failed to refresh cache for server '{name}': {str(e)}")

# Helper function to execute MCP tools
async def execute_mcp_tool(tool_call_name: str, arguments: Dict[str, Any], mcp_manager: MCPManager) -> str:
    """Execute an MCP tool and return the result as a string"""
    timestamp = datetime.now().isoformat()
    
    try:
        # Parse server and tool name from the tool call name (format: server_toolname)
        if '_' in tool_call_name:
            server_name, tool_name = tool_call_name.split('_', 1)
        else:
            # Fallback: try to find the tool in any server
            for server in mcp_manager.get_connected_servers():
                for tool in mcp_manager.tools_cache.get(server, []):
                    if tool['name'] == tool_call_name:
                        server_name = server
                        tool_name = tool_call_name
                        break
            else:
                error_msg = f"Tool '{tool_call_name}' not found in any connected MCP server"
                mcp_logger.error(f"REQUEST | {timestamp} | Tool: {tool_call_name} | Arguments: {json.dumps(arguments)} | Error: Tool not found")
                return error_msg
        
        # Log the request
        mcp_logger.info(f"REQUEST | {timestamp} | Server: {server_name} | Tool: {tool_name} | Arguments: {json.dumps(arguments)}")
        
        result = await mcp_manager.execute_tool(server_name, tool_name, arguments)
        
        if result['success']:
            result_str = str(result['result'])
            # Log the successful response
            mcp_logger.info(f"RESPONSE | {timestamp} | Server: {server_name} | Tool: {tool_name} | Success: True | Result: {result_str[:200]}{'...' if len(result_str) > 200 else ''}")
            return result_str
        else:
            error_msg = f"Tool execution failed: {result.get('error', 'Unknown error')}"
            mcp_logger.error(f"RESPONSE | {timestamp} | Server: {server_name} | Tool: {tool_name} | Success: False | Error: {error_msg}")
            return error_msg
            
    except Exception as e:
        error_msg = f"Error executing tool '{tool_call_name}': {str(e)}"
        mcp_logger.error(f"RESPONSE | {timestamp} | Tool: {tool_call_name} | Exception: {str(e)}")
        return error_msg

def handle_tool_calls_with_openai(client, messages: List[Dict], model: str, tools: List[Dict], mcp_manager: MCPManager) -> str:
    """Handle tool calls with OpenAI in a non-streaming way for better tool execution"""
    try:
        # Make non-streaming call to get tool calls
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        tool_calls = message.tool_calls
        
        if not tool_calls:
            return message.content or ""
        
        # Add assistant's message with tool calls
        messages.append({
            "role": "assistant", 
            "content": message.content,
            "tool_calls": [{"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in tool_calls]
        })
        
        # Execute each tool call
        tool_results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                arguments = {}
            
            # Execute the MCP tool
            result = run_async_in_streamlit(execute_mcp_tool(tool_name, arguments, mcp_manager))
            
            # Add tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })
            # Note: tool_results list removed to hide MCP interactions from chat
        
        # Get final response after tool execution
        final_response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        
        final_content = final_response.choices[0].message.content or ""
        
        # Return only the final response (MCP interactions are now logged separately)
        return final_content
        
    except Exception as e:
        return f"Error handling tool calls: {str(e)}"

# Provider Factory
def get_provider(provider_name: str) -> LLMProvider:
    if provider_name == "OpenAI":
        return OpenAIProvider()
    elif provider_name == "Cohere":
        return CohereProvider()
    else:
        raise ValueError(f"Unknown provider: {provider_name}")

# Set page config with multipage support
st.set_page_config(page_title="Multi-LLM Chat Interface", page_icon="🤖", layout="wide")

# Page navigation
page = st.sidebar.selectbox("Navigate", ["💬 Chat", "📋 MCP Logs"])

if page == "📋 MCP Logs":
    st.title("📋 MCP Interaction Logs")
    
    # Read and display MCP logs
    log_file_path = "mcp_interactions.log"
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Recent MCP Tool Interactions")
    with col2:
        if st.button("🔄 Refresh Logs"):
            st.rerun()
    
    try:
        if os.path.exists(log_file_path):
            with open(log_file_path, 'r') as f:
                logs = f.readlines()
            
            if logs:
                # Display logs in reverse order (newest first)
                st.markdown("### Log Entries (Newest First)")
                
                # Add search/filter functionality
                search_term = st.text_input("🔍 Search logs", placeholder="Filter by server, tool, or content...")
                
                filtered_logs = logs
                if search_term:
                    filtered_logs = [log for log in logs if search_term.lower() in log.lower()]
                
                # Display logs in an expandable container
                with st.container():
                    for i, log_line in enumerate(reversed(filtered_logs[-100:])):
                        log_line = log_line.strip()
                        if log_line:
                            # Parse log entry
                            try:
                                parts = log_line.split(' | ', 3)
                                if len(parts) >= 4:
                                    timestamp = parts[0]
                                    log_level = parts[1]
                                    log_type = parts[2].split(' | ')[0]  # REQUEST or RESPONSE
                                    content = ' | '.join(parts[2:])
                                    
                                    # Color code based on log level and type
                                    if log_level == "ERROR":
                                        st.error(f"**{timestamp}** - {content}")
                                    elif "REQUEST" in log_type:
                                        st.info(f"**{timestamp}** - {content}")
                                    else:  # RESPONSE
                                        st.success(f"**{timestamp}** - {content}")
                                else:
                                    st.text(log_line)
                            except:
                                st.text(log_line)
                            
                            if i < len(filtered_logs) - 1:
                                st.divider()
                
                # Show log statistics
                st.markdown("---")
                st.subheader("📊 Log Statistics")
                
                total_entries = len(logs)
                request_count = len([log for log in logs if "REQUEST" in log])
                response_count = len([log for log in logs if "RESPONSE" in log])
                error_count = len([log for log in logs if "ERROR" in log])
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Entries", total_entries)
                with col2:
                    st.metric("Requests", request_count)
                with col3:
                    st.metric("Responses", response_count)
                with col4:
                    st.metric("Errors", error_count)
                
                # Clear logs button
                st.markdown("---")
                if st.button("🗑️ Clear All Logs", type="secondary"):
                    if st.button("⚠️ Confirm Clear Logs", type="primary"):
                        with open(log_file_path, 'w') as f:
                            f.write('')
                        st.success("Logs cleared successfully!")
                        st.rerun()
            else:
                st.info("No MCP interactions logged yet. Start using MCP tools in the chat to see logs here.")
        else:
            st.info("No log file found. MCP interactions will be logged here once you start using MCP tools.")
            
    except Exception as e:
        st.error(f"Error reading log file: {str(e)}")
    
    # Instructions
    st.markdown("---")
    st.markdown("""
    ### About MCP Logs
    
    This page shows all interactions with MCP (Model Context Protocol) tools, including:
    - **Requests**: When the LLM calls an MCP tool with specific arguments
    - **Responses**: The results returned by MCP tools
    - **Errors**: Any failures during tool execution
    
    Logs are automatically saved to `mcp_interactions.log` and displayed here for easy monitoring and debugging.
    """)
    
else:  # Chat page

    # Title of the application
    st.title("🤖 Multi-LLM Chat Interface")

    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "provider" not in st.session_state:
        st.session_state.provider = "OpenAI"

    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY", "")

    if "cohere_api_key" not in st.session_state:
        st.session_state.cohere_api_key = os.getenv("COHERE_API_KEY", "")

    if "model_name" not in st.session_state:
        st.session_state.model_name = "gpt-4o"

    if "total_cost" not in st.session_state:
        st.session_state.total_cost = 0.0

    # MCP-related session state
    if "mcp_enabled" not in st.session_state:
        st.session_state.mcp_enabled = False

    if "mcp_servers" not in st.session_state:
        st.session_state.mcp_servers = []

    if "mcp_manager" not in st.session_state:
        st.session_state.mcp_manager = MCPManager()

    if "available_tools" not in st.session_state:
        st.session_state.available_tools = []

    # Sidebar for provider, API key and model selection
    with st.sidebar:
        st.header("⚙️ Settings")
    
        # Provider selection
        provider_options = ["OpenAI", "Cohere"]
        provider = st.selectbox("Select LLM Provider", 
                               options=provider_options,
                               index=provider_options.index(st.session_state.provider))
    
        # Get the selected provider instance
        llm_provider = get_provider(provider)
    
        # Dynamic API Key input based on provider
        if provider == "OpenAI":
            api_key = st.text_input("Enter your OpenAI API Key", 
                                   value=st.session_state.openai_api_key, 
                                   type="password")
            st.session_state.openai_api_key = api_key
            current_api_key = st.session_state.openai_api_key
        else:  # Cohere
            api_key = st.text_input("Enter your Cohere API Key", 
                                   value=st.session_state.cohere_api_key, 
                                   type="password")
            st.session_state.cohere_api_key = api_key
            current_api_key = st.session_state.cohere_api_key
    
        # Dynamic model selection based on provider
        model_options = llm_provider.get_models()
        
        # Reset model if switching providers and current model not available
        if st.session_state.model_name not in model_options:
            st.session_state.model_name = model_options[0]
        
        model_name = st.selectbox("Select Model", 
                                 options=model_options, 
                                 index=model_options.index(st.session_state.model_name) if st.session_state.model_name in model_options else 0)
        
        # Update session state
        st.session_state.provider = provider
        st.session_state.model_name = model_name
        
        # Clear messages button
        if st.button("🗑️ Clear Messages"):
            st.session_state.messages = []
            st.session_state.total_cost = 0.0
            st.rerun()
        
        # MCP Configuration Section
        st.markdown("---")
        st.subheader("🔧 MCP Tools")
        
        # MCP Enable/Disable toggle
        mcp_enabled = st.checkbox("Enable MCP Tools", value=st.session_state.mcp_enabled)
        st.session_state.mcp_enabled = mcp_enabled
    
        if mcp_enabled:
            # Add new server section with enhanced configuration
            with st.expander("➕ Add MCP Server", expanded=False):
                # Server addition method selector
                add_method = st.radio(
                    "How would you like to add a server?",
                    options=["Manual Configuration", "Load from File"],
                    horizontal=True,
                    help="Choose to manually configure a server or load from an exported configuration file"
                )
                
                if add_method == "Load from File":
                    # File uploader for loading individual server configs
                    st.subheader("📁 Load Server Configuration")
                    uploaded_file = st.file_uploader(
                        "Choose configuration file",
                        type=['json'],
                        help="Upload a JSON file with exported server configurations",
                        key="individual_config_upload"
                    )
                    
                    if uploaded_file is not None:
                        try:
                            configs_data = json.load(uploaded_file)
                            
                            if not configs_data:
                                st.warning("No server configurations found in file")
                            else:
                                st.success(f"Found {len(configs_data)} server configuration(s)")
                                
                                # Show preview of servers to load
                                with st.expander("📋 Preview Configurations", expanded=True):
                                    for server_name, config_data in configs_data.items():
                                        col1, col2 = st.columns([3, 1])
                                        with col1:
                                            transport_type = config_data.get('transport', 'unknown')
                                            st.write(f"**{server_name}** ({transport_type})")
                                            
                                            # Show brief config details
                                            if transport_type == "stdio":
                                                cmd = config_data.get('config', {}).get('command', 'N/A')
                                                st.caption(f"Command: {cmd}")
                                            elif transport_type in ["http", "websocket"]:
                                                url = config_data.get('config', {}).get('url', 'N/A')
                                                st.caption(f"URL: {url}")
                                        
                                        with col2:
                                            # Check if server already exists
                                            if server_name in st.session_state.mcp_manager.clients:
                                                st.caption("🟡 Exists")
                                            else:
                                                st.caption("🟢 New")
                                
                                if st.button("🚀 Load All Servers", type="primary", key="load_all_from_file"):
                                    loaded_count = 0
                                    failed_count = 0
                                    
                                    with st.spinner("Loading and connecting to servers..."):
                                        for server_name, config_data in configs_data.items():
                                            try:
                                                # Create ServerConfig from loaded data
                                                server_config = ServerConfig.from_dict(config_data)
                                                
                                                # Skip if server already exists
                                                if server_name in st.session_state.mcp_manager.clients:
                                                    st.warning(f"Server '{server_name}' already exists, skipping")
                                                    continue
                                                
                                                # Add server with configuration
                                                success = run_async_in_streamlit(
                                                    st.session_state.mcp_manager.add_server_with_config(server_config)
                                                )
                                                
                                                if success:
                                                    # Update session state
                                                    st.session_state.mcp_servers.append({
                                                        "name": server_name,
                                                        "transport": server_config.transport.value,
                                                        "config": server_config.config,
                                                        "status": "connected"
                                                    })
                                                    loaded_count += 1
                                                else:
                                                    failed_count += 1
                                                    
                                            except Exception as e:
                                                st.warning(f"Failed to load server '{server_name}': {str(e)}")
                                                failed_count += 1
                                    
                                    # Update available tools
                                    st.session_state.available_tools = st.session_state.mcp_manager.get_all_tools()
                                    
                                    # Show results
                                    if loaded_count > 0:
                                        st.success(f"Successfully loaded {loaded_count} server(s)!")
                                    if failed_count > 0:
                                        st.warning(f"Failed to load {failed_count} server(s)")
                                    
                                    st.rerun()
                                        
                        except Exception as e:
                            st.error(f"Failed to parse configuration file: {str(e)}")
                
                else:  # Manual Configuration
                    # Transport protocol selector
                    transport_protocol = st.selectbox(
                        "Transport Protocol",
                        options=[protocol.value for protocol in TransportProtocol],
                        format_func=lambda x: {
                            "stdio": "📡 STDIO (Local Scripts/Commands)",
                            "http": "🌐 HTTP/HTTPS (Remote Servers)",
                            "websocket": "🔌 WebSocket (Real-time)",
                            "server_config": "⚙️ Server Config (Predefined)"
                        }.get(x, x),
                        help="Select the transport protocol for connecting to the MCP server"
                    )
                
                    with st.form("add_mcp_server"):
                        server_name = st.text_input("Server Name", placeholder="e.g., My Local Server")
                        
                        # Protocol-specific configuration
                        if transport_protocol == "stdio":
                            st.subheader("📡 STDIO Configuration")
                            st.info("💡 **For interpreters (python, uvx, node):** Put the script/package as the first argument\n**For direct executables:** Use the executable name as command")
                            
                            command = st.text_input("Command", placeholder="e.g., uvx, python3, node, ./my-server")
                            args_text = st.text_area("Arguments (one per line)", placeholder="mcp-neo4j-cypher@0.3.0\n--transport\nstdio")
                            env_text = st.text_area("Environment Variables (KEY=VALUE, one per line)", placeholder="NEO4J_URI=bolt://localhost:7687\nNEO4J_USERNAME=neo4j\nNEO4J_PASSWORD=password")
                            working_dir = st.text_input("Working Directory (optional)", placeholder="/path/to/server")
                            
                            # Show example configurations
                            with st.expander("📖 Examples", expanded=False):
                                st.code("""# Example 1: uvx with package
Command: uvx
Arguments:
  mcp-neo4j-cypher@0.3.0
  --transport
  stdio

# Example 2: Python script
Command: python3
Arguments:
  my_mcp_server.py
  --port
  8080

# Example 3: Direct executable
Command: ./my-mcp-server
Arguments:
  --config
  config.json""", language="text")
                            
                            # Parse arguments and environment
                            args = [arg.strip() for arg in args_text.split('\n') if arg.strip()] if args_text else []
                            env = {}
                            if env_text:
                                for line in env_text.split('\n'):
                                    if '=' in line:
                                        key, value = line.split('=', 1)
                                        env[key.strip()] = value.strip()
                            
                            config = {
                                "command": command,
                                "args": args,
                                "env": env,
                                "working_dir": working_dir if working_dir else None
                            }
                    
                        elif transport_protocol == "http":
                            st.subheader("🌐 HTTP/HTTPS Configuration")
                            url = st.text_input("Server URL", placeholder="http://localhost:8000/mcp or https://api.example.com/mcp")
                            timeout = st.number_input("Timeout (seconds)", min_value=1, max_value=300, value=30)
                            verify_ssl = st.checkbox("Verify SSL Certificate", value=True)
                            
                            # Headers configuration
                            st.write("**Headers (optional):**")
                            headers_text = st.text_area("Headers (KEY: VALUE, one per line)", placeholder="Authorization: Bearer token123\nContent-Type: application/json")
                            headers = {}
                            if headers_text:
                                for line in headers_text.split('\n'):
                                    if ':' in line:
                                        key, value = line.split(':', 1)
                                        headers[key.strip()] = value.strip()
                            
                            config = {
                                "url": url,
                                "headers": headers,
                                "timeout": timeout,
                                "verify_ssl": verify_ssl
                            }
                        
                        elif transport_protocol == "websocket":
                            st.subheader("🔌 WebSocket Configuration")
                            url = st.text_input("WebSocket URL", placeholder="ws://localhost:8080/mcp or wss://api.example.com/mcp")
                            timeout = st.number_input("Timeout (seconds)", min_value=1, max_value=300, value=30)
                            ping_interval = st.number_input("Ping Interval (seconds)", min_value=1, max_value=120, value=20)
                            
                            # Headers configuration
                            st.write("**Headers (optional):**")
                            headers_text = st.text_area("Headers (KEY: VALUE, one per line)", placeholder="Authorization: Bearer token123")
                            headers = {}
                            if headers_text:
                                for line in headers_text.split('\n'):
                                    if ':' in line:
                                        key, value = line.split(':', 1)
                                        headers[key.strip()] = value.strip()
                            
                            config = {
                                "url": url,
                                "headers": headers,
                                "timeout": timeout,
                                "ping_interval": ping_interval
                            }
                    
                        elif transport_protocol == "server_config":
                            st.subheader("⚙️ Server Config Reference")
                            
                            # Load available predefined servers
                            if 'predefined_servers' not in st.session_state:
                                st.session_state.predefined_servers = run_async_in_streamlit(
                                    st.session_state.mcp_manager.load_predefined_servers()
                                )
                            
                            if st.session_state.predefined_servers:
                                config_name = st.selectbox(
                                    "Select Predefined Server",
                                    options=[server['name'] for server in st.session_state.predefined_servers],
                                    help="Choose from predefined servers in the aggregator config"
                                )
                                
                                # Show selected server details
                                selected_server = next(
                                    (s for s in st.session_state.predefined_servers if s['name'] == config_name), 
                                    None
                                )
                                if selected_server:
                                    st.info(f"**Command:** {selected_server['command']}")
                                    if selected_server['args']:
                                        st.info(f"**Args:** {' '.join(selected_server['args'])}")
                                    if selected_server['env']:
                                        st.info(f"**Environment:** {len(selected_server['env'])} variables")
                            else:
                                config_name = st.text_input("Server Config Name", placeholder="neo4j")
                                st.warning("No predefined servers found. Make sure the aggregator config file exists.")
                            
                            config_path = st.text_input("Config File Path", value="../mcp-aggregator/servers_config.json")
                            
                            config = {
                                "config_name": config_name,
                                "config_path": config_path
                            }
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            test_connection = st.form_submit_button("🔍 Test Connection", type="secondary")
                        with col2:
                            add_server = st.form_submit_button("➕ Add Server", type="primary")
                        
                        # Handle form submissions
                        if test_connection or add_server:
                            if not server_name:
                                st.error("Please provide a server name")
                            elif transport_protocol == "stdio" and not command:
                                st.error("Please provide a command")
                            elif transport_protocol in ["http", "websocket"] and not config.get("url"):
                                st.error("Please provide a URL")
                            elif transport_protocol == "server_config" and not config.get("config_name"):
                                st.error("Please select or provide a config name")
                            else:
                                # Create server config
                                try:
                                    transport_enum = TransportProtocol(transport_protocol)
                                except Exception as e:
                                    st.error(f"❌ Failed to create TransportProtocol enum: {e}")
                                    st.error(f"❌ Available values: {[e.value for e in TransportProtocol]}")
                                    st.stop()
                                
                                server_config = ServerConfig(
                                    name=server_name,
                                    transport=transport_enum,
                                    config=config
                                )
                                
                                if test_connection:
                                    # Test connection only
                                    with st.spinner("Testing connection..."):
                                        try:
                                            success = run_async_in_streamlit(
                                                st.session_state.mcp_manager.test_connection(server_config)
                                            )
                                            if success:
                                                st.success("✅ Connection test successful!")
                                            else:
                                                st.error("❌ Connection test failed")
                                        except Exception as e:
                                            st.error(f"❌ Connection test error: {str(e)}")
                                
                                elif add_server:
                                    # Add server permanently
                                    with st.spinner("Adding server..."):
                                        try:
                                            success = run_async_in_streamlit(
                                                st.session_state.mcp_manager.add_server_with_config(server_config)
                                            )
                                            if success:
                                                # Update session state
                                                st.session_state.mcp_servers.append({
                                                    "name": server_name,
                                                    "transport": transport_protocol,
                                                    "config": config,
                                                    "status": "connected"
                                                })
                                                st.session_state.available_tools = st.session_state.mcp_manager.get_all_tools()
                                                st.success(f"✅ Successfully added server: {server_name}")
                                                st.rerun()
                                            else:
                                                st.error(f"❌ Failed to add server: {server_name}")
                                        except Exception as e:
                                            st.error(f"❌ Error adding server: {str(e)}")
        
            # Display connected servers
            if st.session_state.mcp_servers:
                st.write("**Connected Servers:**")
                for i, server in enumerate(st.session_state.mcp_servers):
                    # Get current connection status from manager
                    current_status = st.session_state.mcp_manager.get_connection_status(server['name'])
                    
                    with st.container():
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            # Status icon with more detail
                            status_icons = {
                                "connected": "🟢",
                                "connection_error": "🔴",
                                "timeout": "🟡",
                                "error": "🔴",
                                "unknown": "⚪"
                            }
                            status_icon = status_icons.get(current_status, "⚪")
                            
                            # Transport protocol icon
                            transport_icons = {
                                "stdio": "📡",
                                "http": "🌐",
                                "websocket": "🔌",
                                "server_config": "⚙️"
                            }
                            transport = server.get('transport', 'unknown')
                            transport_icon = transport_icons.get(transport, "❓")
                            
                            st.write(f"{status_icon} {transport_icon} **{server['name']}**")
                            
                            # Show configuration details
                            if transport == "stdio":
                                config = server.get('config', {})
                                command = config.get('command', 'Unknown')
                                args = config.get('args', [])
                                display_cmd = f"{command} {' '.join(args[:2])}" + ("..." if len(args) > 2 else "")
                                st.caption(f"STDIO: {display_cmd}")
                            elif transport in ["http", "websocket"]:
                                url = server.get('config', {}).get('url', 'Unknown URL')
                                st.caption(f"{transport.upper()}: {url}")
                            elif transport == "server_config":
                                config_name = server.get('config', {}).get('config_name', 'Unknown')
                                st.caption(f"Config: {config_name}")
                            else:
                                # Fallback for old format
                                url = server.get('url', 'Unknown')
                                st.caption(f"Legacy: {url}")
                        
                        with col2:
                            # Connection status badge
                            status_colors = {
                                "connected": "🟢 Connected",
                                "connection_error": "🔴 Error",
                                "timeout": "🟡 Timeout",
                                "error": "🔴 Failed",
                                "unknown": "⚪ Unknown"
                            }
                            st.caption(status_colors.get(current_status, "⚪ Unknown"))
                        
                        with col3:
                            # Action buttons
                            col3a, col3b = st.columns(2)
                            with col3a:
                                if st.button("🔄", key=f"refresh_server_{i}", help="Refresh connection"):
                                    # Refresh server connection
                                    try:
                                        run_async_in_streamlit(
                                            st.session_state.mcp_manager.refresh_tools_cache()
                                        )
                                        st.success(f"Refreshed {server['name']}")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Refresh failed: {str(e)}")
                            
                            with col3b:
                                if st.button("❌", key=f"remove_server_{i}", help="Remove server"):
                                    st.session_state.mcp_manager.remove_server(server['name'])
                                    st.session_state.mcp_servers.pop(i)
                                    st.session_state.available_tools = st.session_state.mcp_manager.get_all_tools()
                                    st.rerun()
                        
                        st.divider()
                
                # Server management actions
                st.markdown("**Server Management:**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📥 Export Configs", help="Export server configurations"):
                        configs = st.session_state.mcp_manager.export_server_configs()
                        if configs:
                            st.json(configs)
                            st.download_button(
                                "💾 Download JSON",
                                data=json.dumps(configs, indent=2),
                                file_name="mcp_server_configs.json",
                                mime="application/json"
                            )
                        else:
                            st.info("No server configurations to export")
                
                with col2:
                    # File uploader for importing configs
                    uploaded_file = st.file_uploader(
                        "📤 Load Configs",
                        type=['json'],
                        help="Load server configurations from exported JSON file"
                    )
                    if uploaded_file is not None:
                        try:
                            configs_data = json.load(uploaded_file)
                            
                            # Load each configuration and connect to servers
                            loaded_count = 0
                            failed_count = 0
                            
                            with st.spinner("Loading and connecting to servers..."):
                                for server_name, config_data in configs_data.items():
                                    try:
                                        # Create ServerConfig from loaded data
                                        server_config = ServerConfig.from_dict(config_data)
                                        
                                        # Skip if server already exists
                                        if server_name in st.session_state.mcp_manager.clients:
                                            st.warning(f"Server '{server_name}' already exists, skipping")
                                            continue
                                        
                                        # Add server with configuration
                                        success = run_async_in_streamlit(
                                            st.session_state.mcp_manager.add_server_with_config(server_config)
                                        )
                                        
                                        if success:
                                            # Update session state
                                            st.session_state.mcp_servers.append({
                                                "name": server_name,
                                                "transport": server_config.transport.value,
                                                "config": server_config.config,
                                                "status": "connected"
                                            })
                                            loaded_count += 1
                                        else:
                                            failed_count += 1
                                            
                                    except Exception as e:
                                        st.warning(f"Failed to load server '{server_name}': {str(e)}")
                                        failed_count += 1
                            
                            # Update available tools
                            st.session_state.available_tools = st.session_state.mcp_manager.get_all_tools()
                            
                            # Show results
                            if loaded_count > 0:
                                st.success(f"Successfully loaded {loaded_count} server configuration(s)!")
                            if failed_count > 0:
                                st.warning(f"Failed to load {failed_count} server configuration(s)")
                            
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Failed to load configurations: {str(e)}")
                
                with col3:
                    if st.button("🔄 Refresh All", help="Refresh all server connections"):
                        with st.spinner("Refreshing all connections..."):
                            try:
                                run_async_in_streamlit(
                                    st.session_state.mcp_manager.refresh_tools_cache()
                                )
                                st.session_state.available_tools = st.session_state.mcp_manager.get_all_tools()
                                st.success("All connections refreshed!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Refresh failed: {str(e)}")
        
            # Display available tools
            if st.session_state.available_tools:
                with st.expander(f"🛠️ Available Tools ({len(st.session_state.available_tools)})", expanded=False):
                    # Group tools by server
                    tools_by_server = {}
                    for tool in st.session_state.available_tools:
                        server_name = tool['server']
                        if server_name not in tools_by_server:
                            tools_by_server[server_name] = []
                        tools_by_server[server_name].append(tool)
                    
                    for server_name, tools in tools_by_server.items():
                        # Get server transport icon
                        server_info = next((s for s in st.session_state.mcp_servers if s['name'] == server_name), None)
                        if server_info:
                            transport = server_info.get('transport', 'unknown')
                            transport_icons = {
                                "stdio": "📡",
                                "http": "🌐", 
                                "websocket": "🔌",
                                "server_config": "⚙️"
                            }
                            transport_icon = transport_icons.get(transport, "❓")
                        else:
                            transport_icon = "❓"
                        
                        st.write(f"**{transport_icon} {server_name}** ({len(tools)} tools)")
                        
                        for tool in tools:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"  🔧 **{tool['name']}**")
                                if 'description' in tool and tool['description']:
                                    st.caption(f"  {tool['description']}")
                            
                            # Show input schema if available
                            if 'inputSchema' in tool and tool['inputSchema']:
                                with col2:
                                    if st.button("📋", key=f"schema_{server_name}_{tool['name']}", help="Show schema"):
                                        st.json(tool['inputSchema'])
                        
                        st.divider()
                    
                    # Tools summary
                    st.markdown("---")
                    total_tools = len(st.session_state.available_tools)
                    total_servers = len(tools_by_server)
                    st.caption(f"📊 **Summary:** {total_tools} tools across {total_servers} servers")
    
        # Show total cost and provider info
        st.markdown("---")
        st.metric("**💰 Total Session Cost**", f"${st.session_state.total_cost:.6f}")
        st.caption(f"Provider: {provider} | Model: {model_name}")
        if st.session_state.mcp_enabled and st.session_state.available_tools:
            st.caption(f"MCP Tools: {len(st.session_state.available_tools)} available")

    # Function to check if current provider is properly configured
    def is_configured():
        if st.session_state.provider == "OpenAI":
            return bool(st.session_state.openai_api_key)
        elif st.session_state.provider == "Cohere":
            return bool(st.session_state.cohere_api_key)
        return False

    # Function to format the message content
    def format_message_content(content):
        if "```" in content:
            # Split by code blocks and render differently
            parts = content.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 0:  # This is regular text
                    if part.strip():
                        st.markdown(part)
                else:  # This is code
                    if '\n' in part:
                        lang_and_code = part.split('\n', 1)
                        if len(lang_and_code) == 2:
                            lang, code = lang_and_code
                            st.code(code, language=lang)
                        else:
                            st.code(part)
                    else:
                        st.code(part)
        else:
            st.markdown(content)


    # Main chat interface
    if not is_configured():
        provider_name = st.session_state.provider
        st.warning(f"⚠️ Please enter your {provider_name} API key in the sidebar to start chatting.")
    else:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                format_message_content(message["content"])
    
        # Chat input
        if prompt := st.chat_input("What would you like to ask?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Display assistant response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                try:
                    # Get current provider and API key
                    provider = get_provider(st.session_state.provider)
                    if st.session_state.provider == "OpenAI":
                        current_api_key = st.session_state.openai_api_key
                    else:
                        current_api_key = st.session_state.cohere_api_key
                    
                    # Validate API key
                    if not current_api_key:
                        raise ValueError(f"No API key provided for {st.session_state.provider}")
                    
                    # Count prompt tokens
                    prompt_tokens = provider.count_tokens(st.session_state.messages, st.session_state.model_name)
                    
                    # Get MCP tools if enabled
                    mcp_tools = []
                    if st.session_state.mcp_enabled and st.session_state.available_tools:
                        mcp_tools = st.session_state.mcp_manager.get_tools_for_llm()
                    
                    # Get chat completion stream
                    stream = provider.chat_completion(
                        messages=st.session_state.messages,
                        model=st.session_state.model_name,
                        api_key=current_api_key,
                        tools=mcp_tools if mcp_tools else None
                    )
                    
                    # Stream the response
                    full_response = ""
                    for chunk in stream:
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                    # Count completion tokens
                    completion_tokens = provider.count_tokens([{"role": "assistant", "content": full_response}], st.session_state.model_name)
                    total_tokens = prompt_tokens + completion_tokens
                    
                    # Cost calculation using provider pricing
                    pricing = provider.get_pricing()
                    model_prices = pricing.get(st.session_state.model_name, {"prompt": 0.01, "completion": 0.03})
                    prompt_cost = (prompt_tokens / 1000) * model_prices["prompt"]
                    completion_cost = (completion_tokens / 1000) * model_prices["completion"]
                    total_cost = prompt_cost + completion_cost
                    
                    # Update total session cost
                    st.session_state.total_cost += total_cost
                    
                    # Display token usage and cost
                    st.markdown("---")
                    st.markdown("### 📊 Token Usage & Cost")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Prompt Tokens", prompt_tokens)
                    with col2:
                        st.metric("Completion Tokens", completion_tokens)
                    with col3:
                        st.metric("Total Tokens", total_tokens)
                    st.metric("**💰 This Query Cost**", f"${total_cost:.6f}")
                
                except Exception as e:
                    error_message = f"❌ Error: {str(e)}"
                    message_placeholder.error(error_message)
                    # Show diagnostic information
                    st.error(f"Error communicating with {st.session_state.provider}. Please check your API key and network connection.")
                    st.code(f"Error type: {type(e).__name__}\nError message: {str(e)}")

    # Footer
    st.markdown("---")