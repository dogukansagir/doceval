import pymupdf as fitz
from google import genai
from google.genai import types
import config

client = genai.Client(api_key=config.GEMINI_API_KEY)

def extract_blocks_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    output = []
    flags = fitz.TEXT_PRESERVE_IMAGES
    for page in doc:
        blocks = page.get_text("blocks", flags=flags)
        for block in blocks:
            if block[6] == 0:  # Text block
                output.append({"type": "text", "content": block[4]})

        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            output.append({"type": "image", "content": base_image["image"], "ext": base_image["ext"]})

    return output

def enrich_blocks(blocks):
    for block in blocks:
        if block["type"] == "image":
            # call vlm model to get the description of the image and add it to the block
            block["content"] = call_vlm_model(block)
            block["type"] = "text"
    return blocks

def call_vlm_model(block):
    response = client.models.generate_content(
    model=config.GEMINI_MODEL,
    contents=[
      types.Part.from_bytes(
        data=block["content"],
        mime_type=f'image/{block["ext"]}',
      ),
      'Extract all data, values, labels and relationships visible in the image. Analyze the data and give me a string about your analysis. you will create strings that are going inside a RAG system. do not omit any information. describe the type and structure of the visual content (table, chart, diagram, figure) before describing its contents.'
        ],
    )

    return response.text