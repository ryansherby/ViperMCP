# server.py

import base64
import os
from typing import Union, List
import io
import re

import torch
import requests

from fastmcp import FastMCP, Context
from fastmcp.utilities.types import Image
from fastmcp.tools.tool import ToolResult

from starlette.responses import Response
from starlette.requests import Request

from pydantic import BaseModel, Field, model_validator

from PIL import Image as PILImage

import viper.src.entrypoint as EntryPoint
from viper.src.model_definitions.openai_model import OpenAIModel

mcp = FastMCP(
    name="ViperMCP",
    stateless_http=True,
)





@mcp.custom_route("/mcp", methods=["GET"])
def smithery_config(request: Request) -> dict:
    """
    Initializes the Smithery configuration for the ViperMCP service.
    
    :return: A boolean indicating that the configuration has been initialized.
    """
    
    openai_api_key = request.query_params.get("apiKey")
    if not openai_api_key:
        return Response(
            "Missing apiKey parameter. Please provide it as a query parameter.",
            status_code=400
        )
    
    OpenAIModel.set_api_key(openai_api_key)
    
    return Response(
        "Query parameters set successfully.",
        status_code=200)



@mcp.custom_route("/health", methods=["GET"])
def health_check(request: Request) -> str:
    """
    Health check endpoint to verify the service is running.
    
    :return: A simple message indicating the service is healthy.
    """
    return Response("OK", status_code=200)



@mcp.custom_route("/device", methods=["GET"])
def get_device(request: Request) -> str:
    """
    Returns the device being used by the ViperAPI.
    
    :return: A string indicating the device (e.g., "cuda" or "cpu").
    """
    return Response(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        status_code=200
    )
    
    

def load_image(data: str) -> PILImage.Image:
    """
    Loads an image from a base64-encoded string or a URL.
    
    :param data: The base64-encoded image data or a URL to the image.
    :return: A PIL Image object.
    """
    if re.match(r"^data:image/.+;base64,", data):
        # Base64 encoded image
        header, encoded = data.split(",", 1)
        return PILImage.open(io.BytesIO(base64.b64decode(encoded)))
    elif data.startswith("http://") or data.startswith("https://"):
        # URL to the image
        response = requests.get(data)
        response.raise_for_status()
        return PILImage.open(io.BytesIO(response.content))
    else:
        # Attempt to decode raw base64 string
        return PILImage.open(io.BytesIO(io.BytesIO(base64.b64decode(data))))


@mcp.tool(description="Returns a text response based on the provided query and image.")
def viper_query(query:str,
                image: str = Field(..., description="URL or base64-encoded image data.")) -> str:
    
    code = EntryPoint.generate_code([query], action=['query'])[0]
    
    image = load_image(image)
    
    res = EntryPoint.execute_code([code], [image])[0]
    
    return res



@mcp.tool(description="Returns a list of images that match the task based on the provided task and image.")
def viper_task(task: str,
                image: str = Field(..., description="URL or base64-encoded image data.")) -> List[Image]:
    
    code = EntryPoint.generate_code([task], action=['task'])[0]
    
    image = load_image(image)
    
    res = EntryPoint.execute_code([code], [image])[0]
    
    if res is None:
        return 
    
    encoded_images = []
    
    for img in res:
        
        pil_img = img.to_pil_image()
        img_bytes = io.BytesIO()
        pil_img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        encoded_images.append(Image(
            data=img_bytes.getvalue(),
            format='png'
        ))
    return encoded_images
        




if __name__ == "__main__":
    mcp.run(
        host="0.0.0.0",
        port=8000,
        transport="streamable-http",
    )