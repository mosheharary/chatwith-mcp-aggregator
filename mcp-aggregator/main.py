import asyncio
import json
import logging
import os
import signal
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Callable, AsyncIterator
import inspect

from fastmcp import FastMCP
from server_manager import McpServerManager




class AppContext:
    """Application context for sharing state across the FastMCP application."""
    def __init__(self, server_manager: McpServerManager, registered_tools: Dict[str, str]):
        self.server_manager = server_manager
        self.registered_tools = registered_tools


class McpAggregatorServer:
    """FastMCP HTTP server that aggregates tools from multiple MCP stdio servers."""
    
    def __init__(self, config_path: str = "servers_config.json"):
        self.config_path = config_path
        
        # Create lifespan context manager
        @asynccontextmanager
        async def app_lifespan(mcp: FastMCP) -> AsyncIterator[AppContext]:
            """Manage application lifecycle."""
            # Startup: Initialize backend servers
            server_manager = await self._initialize_backend_servers()
            registered_tools = {}
            
            try:
                # Register tools from all backend servers
                await self._register_backend_tools(mcp, server_manager, registered_tools)
                
                # Yield context to the application
                yield AppContext(server_manager, registered_tools)
                
            finally:
                # Shutdown: Cleanup backend servers
                await server_manager.stop_all()
                logging.info("Backend MCP servers stopped")
        
        # Create FastMCP instance with lifespan
        self.mcp = FastMCP("MCP Aggregator Server", lifespan=app_lifespan)
        
    async def _initialize_backend_servers(self) -> McpServerManager:
        """Initialize and connect to backend MCP servers."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            server_manager = McpServerManager(config)
            await server_manager.start_all()
            logging.info("Backend MCP servers initialized successfully")
            return server_manager
            
        except Exception as e:
            logging.error(f"Failed to initialize backend servers: {e}")
            raise
            
    async def _register_backend_tools(self, mcp: FastMCP, server_manager: McpServerManager, registered_tools: Dict[str, str]):
        """Dynamically register tools from backend servers as FastMCP tools."""
        try:
            # Get all tools from backend servers
            all_tools = await server_manager.get_all_tools()
            
            for server_name, tools in all_tools.items():
                for tool in tools:
                    tool_name = tool['name']
                    description = tool.get('description', '')
                    input_schema = tool.get('inputSchema', {})
                    
                    # Store mapping for routing
                    registered_tools[tool_name] = server_name
                    
                    # Create proxy function for this tool
                    proxy_func = self._create_tool_proxy(tool_name, server_name, description, server_manager, input_schema)
                    
                    # Register with FastMCP
                    mcp.tool(name=tool_name, description=description)(proxy_func)
                    
                    logging.info(f"Registered tool '{tool_name}' from server '{server_name}'")
                    
        except Exception as e:
            logging.error(f"Error registering backend tools: {e}")
            
    def _create_tool_proxy(self, tool_name: str, server_name: str, description: str, server_manager: McpServerManager, input_schema: Dict[str, Any]) -> Callable:
        """Create a proxy function that routes tool execution to the backend server."""
        
        # Extract parameter names from the input schema
        properties = input_schema.get('properties', {})
        required_params = set(input_schema.get('required', []))
        
        # Create parameter list for the function signature
        params = []
        for param_name, param_info in properties.items():
            if param_name in required_params:
                params.append(f"{param_name}")
            else:
                params.append(f"{param_name}=None")
        
        # Create the function dynamically with explicit parameters
        param_str = ", ".join(params)
        
        # Build kwargs assignment lines
        kwargs_lines = []
        for param in params:
            param_name = param.split("=")[0]
            kwargs_lines.append(f"        if {param_name} is not None: kwargs['{param_name}'] = {param_name}")
        kwargs_assignment = "\n".join(kwargs_lines)
        
        # Build the function code
        func_code = f"""
async def tool_proxy({param_str}) -> Any:
    \"\"\"Proxy function that routes tool execution to backend server.\"\"\"
    import logging
    try:
        # Build kwargs from parameters
        kwargs = {{}}
{kwargs_assignment}
        
        logging.info(f"Executing tool '{tool_name}' on backend server '{server_name}'")
        result = await server_manager.execute_tool(tool_name, kwargs)
        return result
        
    except Exception as e:
        logging.error(f"Error executing tool '{tool_name}': {{e}}")
        raise
"""
        
        # Execute the function code to create the function
        local_vars = {'server_manager': server_manager, 'tool_name': tool_name, 'server_name': server_name, 'Any': Any}
        exec(func_code, local_vars, local_vars)
        tool_proxy = local_vars['tool_proxy']
        
        # Set function metadata for FastMCP
        tool_proxy.__name__ = tool_name
        tool_proxy.__doc__ = description
            
        return tool_proxy
            
    def run(self, host: str = "0.0.0.0", port: int = 8000, path: str = "/mcp"):
        """Run the FastMCP HTTP server."""
        # Run with HTTP transport
        self.mcp.run(
            transport="http",
            host=host,
            port=port,
            path=path
        )


class McpAggregatorServerManager:
    """Manager for running the MCP aggregator server as a daemon."""
    
    def __init__(self, config_path: str = "servers_config.json"):
        self.config_path = config_path
        self.server: Optional[McpAggregatorServer] = None
        
    async def start_server(self, host: str = "0.0.0.0", port: int = 8000, path: str = "/mcp"):
        """Start the aggregator server asynchronously."""
        self.server = McpAggregatorServer(self.config_path)
        
        logging.info(f"Starting MCP Aggregator Server on {host}:{port}{path}")
        
        # Run the server (this will block)
        self.server.run(host=host, port=port, path=path)
        
    async def stop_server(self):
        """Stop the aggregator server gracefully."""
        if self.server:
            await self.server.cleanup()
            
        logging.info("MCP Aggregator Server shutdown complete")


async def main():
    """Main entry point for running the aggregator server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    manager = McpAggregatorServerManager()
    try:
        await manager.start_server()
    except KeyboardInterrupt:
        logging.info("Received interrupt signal")
    except Exception as e:
        logging.error(f"Server error: {e}")
    finally:
        await manager.stop_server()


def run_server_sync():
    """Synchronous entry point for running the server directly."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    server = McpAggregatorServer()
    server.run()


if __name__ == "__main__":
    # For direct execution, use the synchronous entry point
    run_server_sync()
