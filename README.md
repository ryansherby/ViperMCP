# ViperMCP: A Model Context Protocol for Viper Server

ViperMCP is a mixture-of-experts (MoE) visual question-answering (VQA) server that defines several functions to solve 3 particular task areas: 1) visual grounding, 2) compositional image question answering, and 3) external knowledge-dependent image question answering. It is based heavily on the [ViperGPT](https://viper.cs.columbia.edu/) framework.

The MCP server is structured as a [FastMCP](https://gofastmcp.com/getting-started/welcome) streamable-http server and is therefore compatible with all of the client tooling provided by FastMCP.

# Setup

## OpenAI API Key
An [API key for the OpenAI platform](https://platform.openai.com/api-keys) is required. It can either be set in the execution environment as `OPENAI_API_KEY`, referenced by path in the `OPENAI_API_KEY_PATH` environment variable, or passed as an http query parameter.

## Ngrok Account (Optional)
[Ngrok](https://ngrok.com/) can be used to quickly deploy a locally-running server to a public facing URL. Create an account and run `pip install ngrok` to use.

Assuming that you have followed one of the following installation procedures in the next section, running `ngrok http 8000` will forward the public-facing URL to your ViperMCP server.

The address provided by ngrok (or any public facing address) can be used as a substitute for the local address (`http://0.0.0.0:8000`) we will reference below.

# Installation

## Smithery Deployment

ViperMCP can be deployed through [Smithery](https://smithery.ai/docs/build/deployments#custom-deploy).

## Dockerized FastMCP Server

Add your OpenAI API key to a file called `api.key`. In the command below, point the mount 
source to the location of the `api.key`.

```
docker run -i --rm \
--mount type=bind,source=/path/to/api.key,target=/run/secrets/openai_api.key,readonly \
-e OPENAI_API_KEY_PATH=/run/secrets/openai_api.key \
-p 8000:8000 \
rsherby/vipermcp:latest
```

This will begin a CUDA-enabled docker container that can be accessed at `http://0.0.0.0:8000/mcp/`.

Alternatively, you can use the `docker-compose.yaml` file to build the image from source and run it. By default, it assumes that the OpenAI API key can be found in the same directory.

If your container provisioner (e.g., cloud provider) allows you to create environment variables and pass them to the container environment, you can also just set the `OPENAI_API_KEY` variable pre-runtime.

## Pure FastMCP Server

Clone the repository to your local device by running the following comand:
```
git clone --recurse-submodules https://github.com/ryansherby/vipermcp.git
```

After cloning, we need to download the pretrained models and set our OpenAI API key. Run the following commands:
```
cd vipermcp
bash download-models.sh
echo YOUR_OPENAI_API_KEY > api.key
```

We then suggest creating a virtual environment (e.g., conda or venv) and activating it. This is not a requirement but is generally the best practice for managing Python packages. Then, install the requirements by running the follow commands.

```
pip install -r requirements.txt
pip install -e .
```

This will install both the 3rd-party requirements as well the local `viper` package that is used to standardize import locations.

We can now run our local FastMCP server using the follow command.

```
python run_server.py
```

We should be able to access our server now at `http://0.0.0.0:8000/mcp/`.

To utilize the OpenAI related models, we must pass the OpenAI API key to the following URL like:
`http://0.0.0.0:8000/mcp?apiKey=sk-proj-XXXXXXXXXXXXXXXXXXXX`


# Usage

## FastMCP Client

An example with passing base64-encoded byte-level image data. Image URLs can also be passed.
```
async with client:
    await client.ping()

    tools = await client.list_tools()  # Optional

    query = await client.call_tool(
        "viper_query",
        {"query": "how many muffins can each kid have for it to be fair?"},
        {"image": f"data:image/png;base64,{image_base64_string}"}
    )

    task = await client.call_tool(
        "viper_task",
        {"task": "return a mask of all the people in the image"},
        {"image": f"data:image/png;base64,{image_base64_string}"}
    )
```

## OpenAI API

Make sure to send the image URL as "type" : "input_text". Currently, the OpenAI API MCP integration cannot handle byte-level image data, so the image must be sent as a public URL.

```
response = client.responses.create(
    model="gpt-4o",
    tools=[
        {
            "type": "mcp",
            "server_label": "ViperMCP",
            "server_url": f"{server_url}/mcp/",
            "require_approval": "never",
        },
    ],
    input=[
        {
            "role": "system",
            "content": "Forward any queries or tasks relating to an image directly to the ViperMCP server."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "based on this image, how many muffins can each kid have for it to be fair?"
                },
                {
                    "type": "input_text",
                    "text": f"{img_url},
                },
            ],
        },
    ],
)
```

# Appendix

## Models
The following models are used in the default version of ViperMCP:
- Grounding DINO
- SegmentAnything (SAM)
- GPT-4o-mini LLM
- GPT-4o-mini VLM
- GPT-4.1
- X-VLM
- Midas
- BERT

## Warnings
This package generates and executes code on the machine in which it is run. We do not have any direct control over the code that is executed, and thus the prompting mechanism may be used to expose sensitive data. We have included basic injection prevention tools; however, this will not be sufficient to protect your data in a production environment.

If a production-level environment is your goal, we strongly suggest modifying the `src/entrypoint.py` to define separate client wrappers using the same naming convention (i.e., find, simple_query, etc.) that forward requests to a backend server. Then, the `mcp/server.py` should be modified to push requests to this client server, which then makes requests of the backend server. An example flow would be like the following:

```
MCP Server (Query + Image) => Client Server (Generate Code Request) =>
Backend Server (Generates Code) =>
Client Server (Executes Code with Wrapper Functions) =>
Backend Server (Executes Underlying Functions from Wrapper) =>
Client Server (Forwards Result to MCP Server) =>
MCP Server (Returns Result to User)
```

## Citations
Thank you to the team behind ViperGPT! Your framework and subsequent empirical successes have been invaluable in the creation of this project.
```
@article{surismenon2023vipergpt,
    title={ViperGPT: Visual Inference via Python Execution for Reasoning},
    author={D\'idac Sur\'is and Sachit Menon and Carl Vondrick},
    journal={arXiv preprint arXiv:2303.08128},
    year={2023}
}
```

## Contributions

If you'd like to contribute to the project, please pass the necessary tests (found in `/tests`) and create a pull request.



