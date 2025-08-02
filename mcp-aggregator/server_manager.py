import asyncio
import json
import logging
import os
import shutil
import signal
from contextlib import AsyncExitStack
from typing import Any, Dict, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class McpServerProcess:
    """Manages a single MCP server process and its stdio communication."""
    
    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        self.name = name
        self.config = config
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self._process = None
        self._is_healthy = False
        
    async def start(self) -> None:
        """Start the MCP server process and establish connection."""
        # Handle different command types
        command = self.config["command"]
        if command == "npx":
            command = shutil.which("npx")
        elif command == "uvx":
            command = shutil.which("uvx")
        elif command == "python":
            command = shutil.which("python") or shutil.which("python3")
        elif command == "mcp-neo4j-cypher":
            # Check if installed as pip package command
            command = shutil.which("mcp-neo4j-cypher")
        else:
            command = shutil.which(command)
            
        if command is None:
            raise ValueError(f"Command '{self.config['command']}' not found in PATH")
            
        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env={**os.environ, **self.config["env"]}
            if self.config.get("env")
            else None,
        )
        
        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.session = session
            self._is_healthy = True
            logging.info(f"MCP server '{self.name}' started successfully")
            
        except Exception as e:
            logging.error(f"Failed to start MCP server '{self.name}': {e}")
            await self.stop()
            raise
            
    async def stop(self) -> None:
        """Stop the MCP server process and cleanup resources."""
        self._is_healthy = False
        try:
            await self.exit_stack.aclose()
            self.session = None
            logging.info(f"MCP server '{self.name}' stopped")
        except Exception as e:
            logging.error(f"Error stopping MCP server '{self.name}': {e}")
            
    async def health_check(self) -> bool:
        """Check if the server process is healthy."""
        if not self.session:
            return False
            
        try:
            # Simple health check by listing tools
            await self.session.list_tools()
            self._is_healthy = True
            return True
        except Exception as e:
            logging.warning(f"Health check failed for '{self.name}': {e}")
            self._is_healthy = False
            return False
            
    @property
    def is_healthy(self) -> bool:
        """Check if server is currently healthy."""
        return self._is_healthy
        
    async def list_tools(self) -> list[Any]:
        """List available tools from this server."""
        if not self.session:
            raise RuntimeError(f"Server '{self.name}' not initialized")
            
        return await self.session.list_tools()
        
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool on this server."""
        if not self.session:
            raise RuntimeError(f"Server '{self.name}' not initialized")
            
        return await self.session.call_tool(tool_name, arguments)


class McpServerManager:
    """Manages multiple MCP server processes with health monitoring and auto-restart."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.servers: Dict[str, McpServerProcess] = {}
        self._monitoring_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
    async def start_all(self) -> None:
        """Start all configured MCP servers."""
        for name, server_config in self.config.get("mcpServers", {}).items():
            try:
                server = McpServerProcess(name, server_config)
                await server.start()
                self.servers[name] = server
                logging.info(f"Started MCP server: {name}")
            except Exception as e:
                logging.error(f"Failed to start server '{name}': {e}")
                # Continue starting other servers
                
        # Start health monitoring
        self._monitoring_task = asyncio.create_task(self._monitor_servers())
        
    async def stop_all(self) -> None:
        """Stop all MCP servers and cleanup."""
        self._shutdown_event.set()
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
                
        for name, server in self.servers.items():
            try:
                await server.stop()
                logging.info(f"Stopped MCP server: {name}")
            except Exception as e:
                logging.error(f"Error stopping server '{name}': {e}")
                
        self.servers.clear()
        
    async def get_all_tools(self) -> Dict[str, list[Dict[str, Any]]]:
        """Get tools from all healthy servers in a standardized format."""
        all_tools = {}
        for name, server in self.servers.items():
            if server.is_healthy:
                try:
                    tools_response = await server.list_tools()
                    # Convert MCP tools response to standardized format
                    server_tools = []
                    for item in tools_response:
                        if isinstance(item, tuple) and item[0] == "tools":
                            for tool in item[1]:
                                server_tools.append({
                                    "name": tool.name,
                                    "description": tool.description,
                                    "inputSchema": tool.inputSchema,
                                    "title": getattr(tool, 'title', None)
                                })
                    all_tools[name] = server_tools
                except Exception as e:
                    logging.error(f"Error getting tools from '{name}': {e}")
                    all_tools[name] = []
                    
        return all_tools
        
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool on the appropriate server."""
        for name, server in self.servers.items():
            if not server.is_healthy:
                continue
                
            try:
                tools_response = await server.list_tools()
                # Extract tool names from the MCP tools response
                tool_names = []
                for item in tools_response:
                    if isinstance(item, tuple) and item[0] == "tools":
                        tool_names.extend([tool.name for tool in item[1]])
                        
                if tool_name in tool_names:
                    logging.info(f"Executing tool '{tool_name}' on server '{name}'")
                    result = await server.execute_tool(tool_name, arguments)
                    logging.info(f"Tool '{tool_name}' executed successfully")
                    return result
                    
            except Exception as e:
                logging.error(f"Error checking/executing tool on '{name}': {e}")
                
        raise ValueError(f"Tool '{tool_name}' not found on any healthy server")
        
    async def _monitor_servers(self) -> None:
        """Monitor server health and restart failed servers."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                for name, server in list(self.servers.items()):
                    if not await server.health_check():
                        logging.warning(f"Server '{name}' unhealthy, attempting restart")
                        try:
                            await server.stop()
                            await server.start()
                            logging.info(f"Successfully restarted server '{name}'")
                        except Exception as e:
                            logging.error(f"Failed to restart server '{name}': {e}")
                            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in server monitoring: {e}")
                
    def get_server_status(self) -> Dict[str, bool]:
        """Get health status of all servers."""
        return {name: server.is_healthy for name, server in self.servers.items()}