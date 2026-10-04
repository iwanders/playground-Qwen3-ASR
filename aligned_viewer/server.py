#!/usr/bin/env python3
import common  # noqa: I001


import numpy as np 
from aiohttp import web
import aiohttp 
from pathlib import Path
import argparse


import asyncio
 
 
app = web.Application()

THIS_PATH = Path(__file__).parent.absolute()

async def handle(request):
 

    raw_page = "<html><head><title>Aligned files</title></head><body>"
    files = request.app["files"]
    for aligned in files:
        raw_page += f'<a href="/aligned/{aligned}">{aligned}</a><br/>\n'


    
    return web.Response(text=raw_page, content_type = "text/html")
     
async def serve_aligned(request):
    name = request.match_info.get('file', None)
    with open(THIS_PATH/"aligned_view.html") as f:
        raw_page = f.read()
    audio_filename = request.app["files"][name]
    raw_page =  raw_page.replace("this_is_our_awesome_audio_filename.mp3", audio_filename)
    raw_page =  raw_page.replace("this_is_the_aligned_file.json", name)
    return web.Response(text=raw_page, content_type = "text/html")

async def serve_file(request):
    name = request.match_info.get('file', None)
    
    return web.FileResponse(app["directory"] / name)


app.add_routes([web.get('/', handle),
                web.get('/{name}', handle),
                web.get('/aligned/{file}', serve_aligned),
                web.get('/serve_file/{file}', serve_file),
])
app.router.add_static('/assets', path=THIS_PATH / "assets", show_index=True)

def run_server(args): 
    
    #app["pipeline"] = pipeline_worker
    app["directory"] = args.directory
    
    port = args.port
    
    values = []
    
    for d, _, n in args.directory.walk():
        for f in n:
            audio_candidate = f.replace(".json", ".mp3")
            if f.endswith(".json") and audio_candidate in n:
                values.append((f, audio_candidate))
    values = sorted(values)
    app["files"] = dict(values)

    web.run_app(app, port=port)


if __name__ == '__main__':
    # Create a parser with some subcommands
    parser = argparse.ArgumentParser(description="asr server")
    _ = parser.add_argument(
        "-v",
        "--verbose",
        help="Enable verbose output",
        action="store_true",
        default=False,
    ) 
    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="sub-command help")
    parser_run_asr_aligned = subparsers.add_parser("server", help="run asr with alignment")

    parser_run_asr_aligned.add_argument("--port",  type=int,  default=8000, help="Port to bind to." )
    parser_run_asr_aligned.add_argument("directory",  type=Path,  default=8000, help="The directory to host")
    
    parser_run_asr_aligned.set_defaults(func=run_server)



    args = parser.parse_args()
    

    # Execute the selected command's function
    if args.command:
        args.func(args)
    else:
        parser.print_help()
