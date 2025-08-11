from viper.mcp.server import mcp


if __name__ == "__main__":
    mcp.run(
        host="0.0.0.0",
        port=8000,
        transport="streamable-http",
    )