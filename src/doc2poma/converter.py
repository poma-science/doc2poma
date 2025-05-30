import hashlib
import os
import re
import sys
import requests
import base64
import io
import json
from io import StringIO
from io import BytesIO
from urllib.parse import urljoin, urlparse
import subprocess
import shutil
import ctypes.util
import zipfile
import tempfile
import logging

import tiktoken
import litellm
import html2text
import importlib.util
import pandas as pd
import fitz
import pillow_heif
from PIL import Image, ImageSequence
from bs4 import BeautifulSoup
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, before_sleep_log

from poma_senter import clean_and_segment_text


#####################
# Globals and Setup #
#####################


class APIRequestTransientError(Exception):
    """Custom exception for API requests that indicate a transient error and should be retried."""

    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"APIRequestTransientError: {self.message}"


def _load_environment_variables(file_path):
    if not os.path.exists(file_path):
        return
    with open(file_path) as file:
        for line in file:
            if line.strip() and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                if key == "GEMINI_API_KEY":
                    print(f"Setting CONVERTER environment variable: {key}")
                    os.environ[key] = value
                elif key == "OPENAI_API_KEY":
                    print(f"Setting CONVERTER environment variable: {key}")
                    os.environ[key] = value


if "pytest" in sys.modules:
    pass
else:
    _load_environment_variables(".env")


logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
tenacity_logger = logging.getLogger(__name__)
tenacity_logger.setLevel(logging.INFO)


Image.MAX_IMAGE_PIXELS = None
try:
    pillow_heif.register_heif_opener()
except ImportError as error:
    raise Exception(f"Pillow error: {error}")


IMAGE_DESCRIPTION_MAX_OUTPUT_TOKENS = 5000

IMAGE_TYPES_NOT_SUPPORTED_BY_AI_CLIENTS = {"svg", "bmp", "ico", "gif", "tiff", "tif", "heic", "heif"}
SUPPORTED_IMAGE_TYPES = {"png", "jpeg", "jpg", "webp"}
SUPPORTED_INPUT_TYPES = SUPPORTED_IMAGE_TYPES.union({"pdf", "md", "html", "txt"})

SINGLE_IMAGE_PROMPT = (
    "Convert the whole input image to markdown in the detected natural reading order.\n"
    "If there is no title you detect, please prepend an appropriate one.\n"
    "Use inline HTML (not markdown) for *all tables*.\n"
    "Replace all illustrations/photos/graphs/charts by verbal descriptions what is shown.\n"
    "Omit page numbers and decorative watermarks as well as any TOC (table of contents) etc., which are redundant information and not needed.\n"
    "Replace more than three dots (or more than one …) with just ….\n"
    "The language of your answer should be the same as the language of the input image.\n"
    'Do not include any surrounding prose/explanation in your answer (like "Here\'s the markdown conversion" etc.)\n'
    "Do not include any triple backticks, just the raw markdown result."
)
PDF_PAGES_PROMPT = (
    "Convert the whole input (a screenshot of a PDF page) to markdown in the detected natural reading order.\n"
    "Use inline HTML (not markdown) for *all tables*.\n"
    "Omit page numbers and decorative watermarks as well as any TOC (table of contents), which are redundant information and not needed.\n"
    "Replace all illustrations/photos/graphs/charts by verbal descriptions what is shown.\n"
    "Replace more than three dots (or more than one …) with just ….\n"
    "CRITICALLY IMPORTANT: Each sentence MUST be on its own line, separated by a newline character. Do not concatenate multiple sentences into a single line.\n"
    "A sentence typically ends with a period (.), question mark (?), exclamation mark (!), or similar punctuation followed by a space and a capital letter.\n"
    "Do not strictly treat each printed/visual original line as one result line. You should:\n"
    "1. Concatenate line fragments that obviously belong to the same sentence\n"
    "2. But ALWAYS separate different sentences with line breaks\n"
    "The language of your answer should be the same as the language of the input image.\n"
    'Do not include any surrounding prose/explanation in your answer (like "Here\'s the markdown conversion" etc.)\n'
    "Do not include any triple backticks, just the raw markdown result."
)

EMOJI_TABLE = "📋"
TABLE_PLACEHOLDER_PREFIX = f"[{EMOJI_TABLE}TABLE "
TABLE_PLACEHOLDER_SUFFIX = " PLACEHOLDER around here]"

EMOJI_IMAGE = "🖼️"
IMAGE_PLACEHOLDER_PREFIX = f"[{EMOJI_IMAGE}IMAGE "
IMAGE_PLACEHOLDER_SUFFIX = " PLACEHOLDER around here]"

IMAGE_DESCRIPTION_START = "START_IMAGE_DESCRIPTION"
IMAGE_DESCRIPTION_END = "END_IMAGE_DESCRIPTION"

EMOJI_PAGE = "📄"
PAGE_START_DELIMITER_PREFIX = f"[{EMOJI_PAGE}PAGE "
PAGE_START_DELIMITER_SUFFIX = " begins here]"


###############
# Conversions #
###############


def convert(
    full_file_path: str,
    config: dict,
    base_url: str | None,
    debug: bool = False,
) -> tuple[str, float]:
    """
    Convert a document to a .poma archive containing markdown content and metadata.

    Args:
        full_file_path (str): The full path of the input file, e.g. "/home/user/doc.pdf".
        base_url (str | None): The base URL for resolving images in the document.
        model_config (dict): Configuration settings for the conversion model.

    Returns:
        str: The path to the generated .poma archive.
        float: The costs of the conversion in USD, if applicable; 0 if not applicable.
    """

    full_costs: float = 0
    try:
        if not config:
            raise Exception("Configuration for is required for converter.")
        if "conversion_provider" not in config:
            model_config_dump = json.dumps(config, indent=4)
            raise Exception(
                f"Model configuration is incomplete; 'conversion_provider' e.g. 'gemini' is required. Your config: {model_config_dump}"
            )
        if "conversion_model" not in config:
            model_config_dump = json.dumps(config, indent=4)
            raise Exception(
                f"Model configuration is incomplete; 'conversion_model' e.g. 'gemini-2.0-flash' is required. Your config: {model_config_dump}"
            )

        config.update(
            {
                "max_output_tokens": IMAGE_DESCRIPTION_MAX_OUTPUT_TOKENS,
                "temperature": float(config.get("temperature", 0.0)),
                "top_p": float(config.get("top_p", 1.0)),
                "timeout": float(config.get("timeout", 60 * 5)),
                "prompt_single_image": SINGLE_IMAGE_PROMPT,
                "prompt_pdf_pages": PDF_PAGES_PROMPT,
            }
        )

        image_max_size_bytes = 1024 * 1024 * 20
        if config["conversion_provider"] == "mistral":
            image_max_size_bytes = 1024 * 1024 * 10

        if config["conversion_provider"] == "openai":
            SUPPORTED_IMAGE_TYPES.add("heic")
            SUPPORTED_IMAGE_TYPES.add("heif")

        if not base_url:
            print("(doc2poma) WARNING: No base URL provided; images may not be resolved correctly.")

        try:
            filename = os.path.basename(full_file_path)
            input_type = os.path.splitext(filename)[1].lower()[1:]
            input_type = "html" if input_type == "htm" else input_type
            input_type = "jpg" if input_type == "jpeg" else input_type
        except Exception as exception:
            raise Exception(f"Detecting filetype from path '{full_file_path}' failed: {exception}")
        if input_type not in SUPPORTED_INPUT_TYPES:
            raise Exception(
                f"Input type '{input_type}' is not supported. Supported types are: {SUPPORTED_INPUT_TYPES}."
            )

        try:
            with open(full_file_path, "rb") as file:
                input = file.read()
        except Exception as exception:
            raise Exception(f"Reading file at path '{full_file_path}' failed: {exception}")

        try:
            file_id = hashlib.md5(input).hexdigest()
            directory_name = os.path.dirname(full_file_path)
            archive_path = os.path.join(directory_name, f"{file_id}.poma")
        except Exception as exception:
            raise Exception(f"Generating archive path from '{full_file_path}' failed: {exception}")

        html_tables: dict[int, str] = {}
        assets: dict[str, bytes] = {}
        if os.path.exists(archive_path):
            print(f"(doc2poma) NOTE: Skipping conversion. Archive at path already exists: {archive_path}")
        else:
            try:
                if input_type in ("html", "md", "txt"):
                    if isinstance(input, bytes):
                        input = input.decode("utf-8", errors="replace")
                    if input_type == "html":
                        print("(doc2poma) Converting HTML to Markdown…")
                        md_text = _convert_html_to_markdown(input)
                    else:
                        md_text = input
                    md_text = _ensure_html_tables_in_markdown(md_text)
                    print("(doc2poma) Replacing images with descriptions…")
                    md_text, images, images_descr_costs = _replace_images_with_descriptions(
                        md_text, base_url, config, image_max_size_bytes, debug
                    )
                    assets.update(images)
                    full_costs += images_descr_costs
                elif input_type in SUPPORTED_IMAGE_TYPES:
                    print("(doc2poma) Converting image to Markdown…")
                    image_number = 1
                    asset_name = f"image_{image_number:05}.{input_type}"
                    assets[asset_name] = input
                    md_text, img_descr_costs = _convert_image_to_markdown(
                        input, input_type, image_number, config, image_max_size_bytes
                    )
                    md_text = _ensure_html_tables_in_markdown(md_text)
                    full_costs += img_descr_costs
                elif input_type == "pdf":
                    print("(doc2poma) Converting PDF to Markdown…")
                    md_text, pages_as_images, pdf_descr_costs = _convert_pdf_to_markdown(
                        input, config, image_max_size_bytes
                    )
                    assets.update(pages_as_images)
                    md_text = _ensure_html_tables_in_markdown(md_text)
                    full_costs += pdf_descr_costs
                else:
                    raise Exception(
                        f"Input type '{input_type}' is not supported. Supported types are: {SUPPORTED_INPUT_TYPES}."
                    )

                print("(doc2poma) Adding generated title…")
                generated_title, title_costs = _generate_title_from_markdown(md_text, config)
                md_text = f"# {generated_title}\n\n{md_text}" if generated_title else md_text
                full_costs += title_costs

                print("(doc2poma) Replacing tables with placeholders…")
                md_text, html_tables = _replace_tables_with_placeholders(md_text)

            except Exception as exception:
                raise Exception(f"Converting file '{filename}' failed: {exception}")

            if base_url:
                try:
                    print("(doc2poma) Adding base_url to relative links…")

                    def replace_link(match):
                        text, href = match.groups()
                        if not href.startswith(("http://", "https://", "mailto:", "tel:", "#", "ftp://")):
                            href = urljoin(base_url, href)
                        return f"{text}({href})"

                    pattern = re.compile(r"(\[.*?\])\((.*?)\)")
                    md_text = pattern.sub(replace_link, md_text)
                except Exception as exception:
                    raise Exception(f"Adding base_url to relative links failed: {exception}")

            try:
                print("(doc2poma) Cleaning and splitting Markdown…")
                md_text = clean_and_segment_text(md_text)
                md_text = _strip_empty_and_br_lines(md_text)
            except Exception as exception:
                raise Exception(f"Cleaning and splitting Markdown failed: {exception}")

            print("(doc2poma) Generating archive file…")
            temp_dir = tempfile.mkdtemp(prefix="poma_zip_")

            content_path = os.path.join(temp_dir, "content.md")
            _file_write(content_path, md_text, encoding="utf-8")

            tables_directory = os.path.join(temp_dir, "tables")
            _create_directory_if_not_exists(tables_directory)
            if html_tables:
                for table_number, html_table in html_tables.items():
                    table_path = os.path.join(tables_directory, f"table_{table_number:05}.html")
                    _file_write(table_path, html_table, encoding="utf-8")

            assets_directory = os.path.join(temp_dir, "assets")
            _create_directory_if_not_exists(assets_directory)
            if assets:
                for asset_name, asset_data in assets.items():
                    asset_path = os.path.join(assets_directory, asset_name)
                    _file_write(asset_path, asset_data)

            try:
                with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                    zipf.write(content_path, arcname="content.md")
                    zipf.comment = json.dumps(
                        {
                            "source_file": filename,
                            "source_type": input_type,
                            "creation_date": pd.Timestamp.now().isoformat(),
                            "version": "1.0",
                        }
                    ).encode()
                    if html_tables and os.path.exists(tables_directory):
                        for filename in os.listdir(tables_directory):
                            file_path = os.path.join(tables_directory, filename)
                            if os.path.isfile(file_path):
                                zipf.write(file_path, arcname=f"tables/{filename}")
                    if assets and os.path.exists(assets_directory):
                        for filename in os.listdir(assets_directory):
                            file_path = os.path.join(assets_directory, filename)
                            if os.path.isfile(file_path):
                                zipf.write(file_path, arcname=f"assets/{filename}")
            except Exception as exception:
                raise Exception(f"Creating archive '{archive_path}' failed: {exception}")
            finally:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)

        return archive_path, full_costs

    except Exception as exception:
        raise Exception(f"(doc2poma) ERROR: {exception}")


def _convert_image_to_markdown(
    image_data: bytes,
    image_type: str,
    image_number: int,
    conversion_params: dict,
    image_max_size_bytes: int,
) -> tuple[str, float]:
    """
    Convert an image to markdown format using AI for description.
    The image is resized to fit within the specified maximum size before being sent to the AI model.

    Args:
        image_data (bytes): The image data in bytes.
        image_type (str): The type of the image (e.g. 'png').
        image_number (int): The number of the image in the document, used for placeholder generation.
        conversion_params (dict): Parameters for AI conversion including provider, model, temperature, max_output_tokens, and top_p.
        image_max_size_bytes (int): The maximum size in bytes for the image.

    Returns:
        tuple[str, float]: A tuple containing the generated description of the image (markdown representation)
        and the cost in USD of generating the description.
        The description is wrapped in delimiters for easy identification.
    """

    def resize_image_to_max_size(image_data: bytes, filetype: str, max_size_bytes: int) -> bytes:
        input_buffer = io.BytesIO(image_data)
        image = Image.open(input_buffer)
        image_format = image.format or filetype
        if not image_format:
            raise Exception("format could not be determined")
        output_buffer, final_size_bytes = _resize_image(image, image_format, max_size_bytes)
        print(f"(doc2poma) Sucessfully resized image to {final_size_bytes} bytes.")
        return output_buffer.read()

    def image_bytes_to_base64_data_uri(image_data, format: str):
        b64 = base64.b64encode(image_data).decode("ascii")
        return f"data:image/{format};base64,{b64}"

    try:
        original_size_bytes = _get_size_of_image_data(image_data, image_type)
        if original_size_bytes < 0:
            raise Exception(f"Converting image to Markdown failed: size could not be determined")
        if round(original_size_bytes, 5) > round(image_max_size_bytes, 5):
            print(
                f"(doc2poma) Image exceeds maximum size of {image_max_size_bytes} bytes: {original_size_bytes}. Resizing…"
            )
            try:
                image_data = resize_image_to_max_size(image_data, image_type, image_max_size_bytes)
            except Exception as exception:
                raise Exception(
                    f"Error resizing image from {original_size_bytes} bytes to maximum {image_max_size_bytes} bytes: {exception}"
                )

        data_uri = image_bytes_to_base64_data_uri(image_data, image_type)

        print("(doc2poma) Sending image to AI for description…")
        provider = conversion_params["conversion_provider"]
        model = conversion_params["conversion_model"]
        prompt = conversion_params["prompt_single_image"]
        litellm.drop_params = True
        response = litellm.completion(
            model=f"{provider}/{model}",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
            timeout=float(conversion_params.get("timeout", 60 * 5)),
            temperature=float(conversion_params["temperature"]),
            max_completion_tokens=int(conversion_params["max_output_tokens"]),
            top_p=float(conversion_params["top_p"]),
        )
        description = str(response.get("choices", [{}])[0].get("message", {}).get("content", ""))
        cost_usd = response._hidden_params.get("response_cost", 0.0)
        if not cost_usd or not isinstance(cost_usd, (int, float)):
            print(
                f"(doc2poma) WARNING: No cost information for describing image {image_number} - total cost estimated will be inaccurate."
            )

        if not description:
            raise APIRequestTransientError("API returned empty description for image.")

        print("(doc2poma) Successfully generated description for image.")
        placeholder = f"{IMAGE_PLACEHOLDER_PREFIX}{image_number}{IMAGE_PLACEHOLDER_SUFFIX}"
        delimiter = (
            f"\n\n{placeholder} {IMAGE_DESCRIPTION_START} {description.strip()} {IMAGE_DESCRIPTION_END}\n\n"
        )
        return delimiter, cost_usd
    except (
        litellm.ServiceUnavailableError,
        litellm.APIConnectionError,
        litellm.RateLimitError,
        litellm.BadRequestError,
    ) as exception:
        # Re-raise a custom transient exception to trigger retry
        raise APIRequestTransientError(f"API connection/service/rate-limit error: {exception}") from exception
    except Exception as exception:
        # Do not retry
        raise Exception(f"Error getting description for {image_type}-image: {exception}")


@retry(
    wait=wait_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(
        (
            litellm.ServiceUnavailableError,
            litellm.APIConnectionError,
            litellm.RateLimitError,
            litellm.BadRequestError,
            APIRequestTransientError,
        )
    ),
    before_sleep=before_sleep_log(tenacity_logger, logging.INFO, exc_info=True),
)
def _convert_pdf_page_to_markdown(
    image: Image.Image,
    image_type: str,
    conversion_params: dict,
    image_max_size_bytes: int,
    is_first_and_only_page: bool,
    page_number: int,
) -> tuple[str, float]:
    """
    Convert a PDF page image to markdown format by sending it to an AI model for description.

    Args:
        image (PIL.Image): The PDF page image to convert.
        image_type (str): The type of the image ('png').
        conversion_params (dict): Parameters for AI conversion including provider, model, temperature, max_output_tokens, and top_p.
        image_max_size_bytes (int): The maximum size in bytes for the image.
        is_first_and_only_page (bool): Flag indicating if it's the first and only page.
        page_number (int): The page number of the PDF page, used for placeholder generation.

    Returns:
        tuple[str, float]: A tuple containing the generated description of the PDF page (markdown representation)
        and the cost in USD of generating the description.
        Each page description is prepended with a delimiter for easy identification.
    """

    def pil_image_to_base64_data_uri(img: Image.Image, format="PNG", **save_kwargs) -> str:
        """Convert a PIL Image to a base64-encoded data URI."""
        buffer = io.BytesIO()
        img.save(buffer, format=format, **save_kwargs)
        b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/{format.lower()};base64,{b64}"

    try:
        original_size_bytes, _ = _get_size_and_buffer_of_image(image, image_type, quality=100)
        if round(original_size_bytes, 5) > round(image_max_size_bytes, 5):
            print(
                f"(doc2poma) Page-image {page_number} exceeds maximum size of {image_max_size_bytes} bytes: {original_size_bytes}. Resizing…"
            )
            try:
                image_format = image.format or image_type
                if not image_format:
                    raise Exception("format could not be determined")
                output_buffer, final_size_bytes = _resize_image(image, image_type, image_max_size_bytes)
                image = Image.open(output_buffer)
                image.format = image_format
                print(f"(doc2poma) Sucessfully resized page-image {page_number} to {final_size_bytes} bytes.")
            except Exception as exception:
                raise Exception(
                    f"Error resizing page-image {page_number} from {original_size_bytes} bytes to maximum {image_max_size_bytes} bytes: {exception}"
                )

        data_uri = pil_image_to_base64_data_uri(image, image_type)

        print(f"(doc2poma) Sending page {page_number} to AI for description…")
        provider = conversion_params["conversion_provider"]
        model = conversion_params["conversion_model"]
        prompt = (
            conversion_params["prompt_single_image"]
            if is_first_and_only_page
            else conversion_params["prompt_pdf_pages"]
        )
        litellm.drop_params = True
        response = litellm.completion(
            model=f"{provider}/{model}",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
            timeout=float(conversion_params.get("timeout", 60 * 5)),
            temperature=float(conversion_params["temperature"]),
            max_completion_tokens=int(conversion_params["max_output_tokens"]),
            top_p=float(conversion_params["top_p"]),
        )
        description = str(response.get("choices", [{}])[0].get("message", {}).get("content", ""))
        cost_usd = response._hidden_params.get("response_cost", 0.0)
        if not cost_usd or not isinstance(cost_usd, (int, float)):
            print(
                f"(doc2poma) WARNING: No cost information for describing page {page_number} - total cost estimated will be inaccurate."
            )

        if not description:
            raise APIRequestTransientError("API returned empty description for PDF page.")

        print(f"(doc2poma) Successfully generated description for page {page_number}.")
        delimiter = f"\n{PAGE_START_DELIMITER_PREFIX}{page_number}{PAGE_START_DELIMITER_SUFFIX}\n\n{description.strip()}\n\n"
        return delimiter, cost_usd
    except (
        litellm.ServiceUnavailableError,
        litellm.APIConnectionError,
        litellm.RateLimitError,
        litellm.BadRequestError,
    ) as exception:
        # Re-raise as custom transient exception to trigger retry
        raise APIRequestTransientError(
            f"API connection/service/rate-limit error for page {page_number}: {exception}"
        ) from exception
    except Exception as exception:
        # Do not retry
        raise Exception(f"Error getting description for pdf-page {page_number}: {exception}")


def _convert_pdf_to_markdown(
    pdf_data: bytes,
    conversion_params: dict,
    image_max_size_bytes: int,
) -> tuple[str, dict[str, bytes], float]:
    """
    Convert a PDF document to markdown format by sending each page to an AI model for description.
    This method handles the resizing of images to ensure they fit within the specified maximum size.

    Args:
        pdf_data (bytes): The binary data of the PDF document.
        conversion_params (dict): Parameters for the conversion process, including AI model details.
        image_max_size_bytes (int): The maximum size in bytes for the images.

    Returns:
        tuple[str, dict[str, bytes], float]:
        A tuple containing the markdown content generated from the descriptions provided by the AI model
        for each page (separated by delimiters for easy identification),
        a dictionary mapping page image names to their bytes representations,
        and the total cost in USD of generating the descriptions.
    """

    try:
        doc = fitz.open("pdf", pdf_data)
    except Exception as exception:
        raise Exception(f"Error loading PDF: {exception}")

    pages_count = len(doc)
    pages: dict[str, bytes] = {}
    descriptions_per_page: dict[int, str] = {}
    errors_per_page: dict[int, str] = {}
    pdf_descr_costs: float = 0

    print(f"(doc2poma) Processing {pages_count} pdf-pages…")
    for index in range(pages_count):
        page_number = index + 1
        image_type = "png"
        try:
            page = doc.load_page(index)
            pix = page.get_pixmap(dpi=150)
            if pix.n == 1:
                mode = "L"  # grayscale
            elif pix.n == 3:
                mode = "RGB"  # truecolor
            elif pix.n == 4:
                mode = "RGBA"  # truecolor + alpha
            else:
                raise ValueError(f"unsupported pixmap format: {pix.n} channels")
            image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            image_name = f"page_{page_number:05}.{image_type}"
            temp_buffer = io.BytesIO()
            image.save(temp_buffer, format=image_type.upper())
            pages[image_name] = temp_buffer.getvalue()
        except Exception as exception:
            errors_per_page[page_number] = f"Image extraction error: {exception}"
            continue

        try:
            description, page_descr_costs = _convert_pdf_page_to_markdown(
                image,
                image_type,
                conversion_params,
                image_max_size_bytes,
                is_first_and_only_page=(index == 0 and pages_count == 1),
                page_number=page_number,
            )
            descriptions_per_page[page_number] = description
            pdf_descr_costs += page_descr_costs
        except APIRequestTransientError as exception:
            errors_per_page[page_number] = (
                f"AI request failed after retries for page {page_number}: {exception}"
            )
        except Exception as exception:
            errors_per_page[page_number] = (
                f"Unexpected error during AI conversion for page {page_number}: {exception}"
            )

    doc.close()

    print(f"(doc2poma) Generated descriptions for {len(descriptions_per_page)}/{pages_count} pages.")
    if errors_per_page or len(descriptions_per_page) < pages_count:
        errors_dump = json.dumps(errors_per_page, indent=4)
        raise Exception(f"Error getting descriptions for pdf-pages. Details: {errors_dump}")

    def stitch_markdown_responses(responses_per_page: dict[int, str]) -> str:
        sorted_pages = sorted(responses_per_page.items())
        if not sorted_pages:
            return ""
        full_markdown = sorted_pages[0][1]  # Start with first page
        for _, response_text in sorted_pages[1:]:
            full_markdown += response_text
        return full_markdown

    markdown_content = stitch_markdown_responses(descriptions_per_page)
    return markdown_content, pages, pdf_descr_costs


def _convert_html_to_markdown(html: str | bytes) -> str:
    """
    Convert HTML content to Markdown format.
    The method protects specific HTML tags and inline LaTeX, Handlebars,
    and Jinja-style templates from being converted to Markdown.
    It also ensures all images are properly extracted and converted to markdown format.

    Args:
        html (str | bytes): The HTML content to be converted to Markdown.

    Returns:
        str: The Markdown formatted content.
    """

    try:
        if isinstance(html, bytes):
            try:
                html = html.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    html = html.decode("latin-1")
                except Exception as exception:
                    raise Exception(f"Failed to decode HTML content: {exception}")

        soup = BeautifulSoup(html, "html.parser")
        protected = []
        image_placeholders = []

        # Process all links that contain images, separating image links into standalone images and links.
        # For example: <a href="link-url"><img src="image-url" alt="Alt Text"></a>
        # becomes: <img src="image-url" alt="Alt Text"> <a href="link-url">link</a>
        for a_tag in soup.find_all("a"):
            img_tag = a_tag.find("img")
            if img_tag:
                img_clone = img_tag.extract()
                link_href = a_tag.get("href")
                link_text = f"(Link: {link_href})"
                a_tag.string = link_text
                a_tag["href"] = link_href
                a_tag.insert_before(img_clone)
                a_tag.insert_before(" ")

        # Process all images to ensure they're properly converted to markdown
        for img_tag in soup.find_all("img"):
            alt_text = img_tag.get("alt", "")
            src = img_tag.get("src", "")
            if src:
                md_image = f"![{alt_text}]({src})"
                image_placeholders.append((str(img_tag), md_image))

        def flatten_tables(html_or_soup: str | BeautifulSoup) -> BeautifulSoup:
            soup = (
                html_or_soup
                if isinstance(html_or_soup, BeautifulSoup)
                else BeautifulSoup(html_or_soup, "lxml")
            )
            while True:
                inners = soup.select("table table")  # any <table> with a table ancestor
                if not inners:
                    break
                for inner in inners:
                    inner.unwrap()  # drop the tag, keep its children
            return soup

        soup = flatten_tables(html)

        # Protect specific HTML tags
        tags_to_protect = ["table", "pre", "code", "script"]

        def protect_tag(tag):
            protected.append(str(tag))
            tag.replace_with(f"PROTECTED_BLOCK_{len(protected) - 1}")

        # Protect all tags in protection list
        for tag_name in tags_to_protect:
            for tag in soup.find_all(tag_name):
                protect_tag(tag)

        # Handle text-based blocks from the raw HTML string (not parsed)

        raw_html = str(soup)

        # Triple backtick code blocks
        def protect_code_block(match):
            protected.append(match.group(0))
            return f"PROTECTED_BLOCK_{len(protected) - 1}"

        raw_html = re.sub(r"```[\s\S]*?```", protect_code_block, raw_html)

        # Inline LaTeX ($...$ and $$...$$)
        raw_html = re.sub(r"\${1,2}.*?\${1,2}", protect_code_block, raw_html)

        # Handlebars / Jinja-style templates
        raw_html = re.sub(r"\{\{.*?\}\}", protect_code_block, raw_html)

        # Convert remaining HTML to Markdown
        h = html2text.HTML2Text()
        h.body_width = 0
        h.use_automatic_links = True
        h.ignore_images = False
        h.ignore_links = False
        h.ignore_emphasis = False
        h.heading_style = "ATX"

        md = h.handle(raw_html)

        # Restore everything
        for index, block in enumerate(protected):
            md = md.replace(f"PROTECTED_BLOCK_{index}", block)

        def unescape_markdown_artifacts(md: str) -> str:
            # Undo over-escaped list numbers and punctuation
            return re.sub(r"(?<=\d)\\\\.", ".", md).replace("\\*", "*").replace("\\_", "_")

        md = unescape_markdown_artifacts(md)

        # Verify again all images are properly converted to markdown
        # Some images might be missed by html2text, especially those with complex attributes
        for html_img, md_img in image_placeholders:
            # Check if the markdown already contains this image
            if md_img not in md and html_img not in md:
                md += f"\n\n{md_img}\n"
                print(f"(doc2poma) Added missing image: {md_img}")

    except Exception as exception:
        raise Exception(f"Error converting HTML to Markdown: {exception}")

    return md


def _ensure_html_tables_in_markdown(md_text: str) -> str:
    """
    Ensure that Markdown tables in the input text are converted to inline HTML tables.

    Args:
        md_text (str): The input text containing Markdown tables.

    Returns:
        str: The text with Markdown tables converted to inline HTML tables.
    """

    def replacer(match):
        md_table = match.group(1)
        lines = md_table.strip().split("\n")
        # Remove the second line (separator row: |---|---|)
        clean_lines = [line for i, line in enumerate(lines) if i != 1]
        # Convert Markdown table to CSV format for Pandas
        csv_like = "\n".join(clean_lines).replace("|", ",")
        df = pd.read_csv(StringIO(csv_like))
        # Convert DataFrame to inline HTML
        return df.to_html(index=False, border=1, escape=False)

    try:
        table_pattern = re.compile(r"(^\|.*\n(\|[-:]+)+\n(\|.*\n?)+)", re.MULTILINE)
        return table_pattern.sub(replacer, md_text)
    except Exception as exception:
        raise Exception(f"Error converting Markdown tables to HTML: {exception}")


def _clean_html_table(raw_html: str) -> str:
    """
    Cleans up an HTML table by removing unwanted tags and attributes, and returns the cleaned HTML table.

    Args:
        raw_html (str): The raw HTML content of the table.

    Returns:
        str: The cleaned HTML table content.
    """
    try:
        soup = BeautifulSoup(raw_html, "html.parser")
        for tag in soup.find_all(True):
            if tag.name in ["script", "style"]:
                tag.decompose()  # Remove script and style tags entirely
                continue
            if tag.name not in ["table", "thead", "tbody", "tfoot", "tr", "th", "td"]:
                tag.unwrap()  # Remove unwanted tags but keep contents
                continue
            # Remove all attributes except 'rowspan' and 'colspan' for <td>/<th>
            allowed_attrs = ["rowspan", "colspan"] if tag.name in ["td", "th"] else []
            tag.attrs = {key: value for key, value in tag.attrs.items() if key in allowed_attrs}

        cleaned_lines = []
        for line in str(soup).splitlines():
            line = line.lstrip()
            line = line.replace("\r", "")
            line = line.replace("\n", "")
            if not line:
                continue
            cleaned_lines.append(line)
        cleaned_html = "".join(cleaned_lines)
        return cleaned_html

    except Exception as exception:
        raise Exception(f"Error cleaning HTML table: {exception}")


def _replace_tables_with_placeholders(md_text: str) -> tuple[str, dict[int, str]]:
    """
    Replace tables with placeholders in the markdown content.

    Args:
        md_text (str): The markdown text containing tables.

    Returns:
        tuple[str, dict[int, str]]: A tuple containing the modified markdown content with placeholders for tables
        and a dictionary mapping table numbers to their cleaned HTML representations.
    """

    saved_tables = {}
    success_count = 0
    table_pattern = re.compile(
        r"<table\b[^>]*>.*?</table>|<div\s+class=['\"]navbox[^>]*>.*?</div>|<div\s+class=['\"]infobox[^>]*>.*?</div>",
        re.DOTALL | re.IGNORECASE,
    )
    try:
        html_tables = table_pattern.findall(md_text)
        print(f"(doc2poma) Processing {len(html_tables)} tables…")
        output_md = md_text
        for table_number, html_table in enumerate(html_tables, start=1):
            clean_table = _clean_html_table(html_table)
            saved_tables[table_number] = clean_table
            placeholder = f"\n\n{TABLE_PLACEHOLDER_PREFIX}{table_number}{TABLE_PLACEHOLDER_SUFFIX}\n\n"
            output_md = output_md.replace(str(html_table), placeholder, 1)
            success_count += 1
        print(f"(doc2poma) Replaced {success_count}/{len(html_tables)} tables with placeholders.")
    except Exception as exception:
        raise Exception(f"Replacing tables in Markdown failed: {exception}")
    return output_md, saved_tables


def _replace_images_with_descriptions(
    md_text: str,
    base_url: str | None,
    conversion_params: dict,
    image_max_size_bytes: int,
    debug: bool = False,
) -> tuple[str, dict[str, bytes], float]:
    """
    Process images in the markdown content, replace them with descriptions.

    Args:
        md_text (str): The markdown text containing images.
        base_url (str | None): The base URL for relative image URLs.
        conversion_params (dict): Parameters for image conversion.
        image_max_size_bytes (int): Maximum size in bytes for images.

    Returns:
        tuple[str, dict[str, bytes]]:
        A tuple containing the output markdown content with images
        replaced by descriptions (separated by delimiters for easy identification),
        a dictionary mapping image names to their bytes representations,
        and the total cost in USD of generating the descriptions.
    """

    def is_relative_url(url: str) -> bool:
        if not url or not isinstance(url, str):
            return False
        if url.startswith("//"):
            return True
        parsed_url = urlparse(url)
        return not parsed_url.scheme and not parsed_url.netloc and bool(parsed_url.path)

    def is_wikipedia_image_url(url: str) -> bool:
        if not url or not isinstance(url, str):
            return False
        parsed_url = urlparse(url)
        if not parsed_url.netloc or not (
            parsed_url.netloc.endswith("wikipedia.org") or parsed_url.netloc.endswith("wikimedia.org")
        ):
            return False
        if "/wiki/" not in parsed_url.path:
            return False
        path_parts = parsed_url.path.split("/")
        for part in path_parts:
            if ":" in part and any(ext in part.lower() for ext in [".jpg", ".jpeg", ".png", ".gif", ".svg"]):
                return True
        return False

    def get_wikipedia_image_url(url: str) -> str:
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            path_parts = parsed_url.path.split("/")
            file_name = None
            for part in path_parts:
                if ":" in part:
                    file_name = part
                    break
            if not file_name:
                return url

            # Construct API URL - use 800px as target size for better AI vision processing
            # The API will maintain aspect ratio and set the max dimension to this value
            api_url = f"https://{domain}/w/api.php?action=query&titles={file_name}&prop=imageinfo&iiprop=url&iiurlwidth=800&format=json"
            response = requests.get(api_url)
            if response.status_code != 200:
                return url

            data = response.json()
            pages = data.get("query", {}).get("pages", {})
            if not pages:
                return url

            page = next(iter(pages.values()))
            image_info = page.get("imageinfo", [])
            if not image_info:
                return url

            thumb_url = image_info[0].get("thumburl")
            if thumb_url:
                return thumb_url

            return image_info[0].get("url", url)

        except Exception as exception:
            print(f"(doc2poma) WARNING: Failed to get Wikipedia image URL: {exception}")
            return url

    def is_base64_image(image_src: str) -> bool:
        return image_src.startswith("data:image") and "base64" in image_src

    def image_extension_from_content_type(content_type: str) -> str | None:
        if not content_type:
            return None
        if not content_type.startswith("image/"):
            return None
        parts = content_type.split("/")
        if len(parts) != 2:
            return None
        second_part = parts[1]
        if "+" in second_part:
            more_parts = second_part.split("+")
            return more_parts[0]
        else:
            return second_part

    def extract_type_and_data_from_image_src(
        image_src: str,
    ) -> tuple[str | None, bytes | None]:
        try:
            parts = image_src.split(",")
            content_type = parts[0].split(";")[0].split(":")[1]
            image_type = image_extension_from_content_type(content_type)
            image_data = base64.b64decode(parts[1])
            return image_type, image_data
        except Exception as exception:
            return None, None

    def download_type_and_data_from_url(url: str) -> tuple[str | None, bytes | None]:
        try:
            response = requests.get(
                url,
                timeout=10,
                headers={
                    # Mimic a Safari browser request to prevent 403 errors
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.5 Safari/605.1.15",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Sec-Fetch-Site": "none",
                    "Sec-Fetch-Mode": "navigate",
                    "Sec-Fetch-Dest": "document",
                    # Don't send a Referer header as it can trigger anti-scraping measures
                    "Referer": "",
                },
            )

            if response.status_code != 200:
                print(
                    f"(doc2poma) WARNING: Failed to download image from {url}: HTTP status {response.status_code}"
                )
                return None, None

            content_type = response.headers.get("content-type")
            if not content_type or not content_type.startswith("image/"):
                print(
                    f"(doc2poma) WARNING: Content from {url} is not an image (content-type: {content_type})"
                )
                return None, None

            image_type = image_extension_from_content_type(content_type)
            if not image_type:
                print(f"(doc2poma) WARNING: Could not determine image type from content-type: {content_type}")
                if url.lower().endswith(".jpg") or url.lower().endswith(".jpeg"):
                    image_type = "jpeg"
                elif url.lower().endswith(".png"):
                    image_type = "png"
                elif url.lower().endswith(".gif"):
                    image_type = "gif"
                elif url.lower().endswith(".svg"):
                    image_type = "svg"
                elif url.lower().endswith(".webp"):
                    image_type = "webp"

            return image_type, response.content
        except requests.exceptions.Timeout:
            print(f"(doc2poma) WARNING: Timeout while downloading image from {url}")
            return None, None
        except requests.exceptions.SSLError as exception:
            print(f"(doc2poma) WARNING: SSL error while downloading image from {url}: {exception}")
            return None, None
        except requests.exceptions.RequestException as exception:
            print(f"(doc2poma) WARNING: Request error while downloading image from {url}: {exception}")
            return None, None
        except Exception as exception:
            print(f"(doc2poma) WARNING: Unexpected error while downloading image from {url}: {exception}")
            return None, None

    def get_fallback_text_in_md_format(alt_text, title, image_src: str | None) -> str:
        fallback_text = alt_text or title or "IMG"
        if not image_src:
            return f"![{fallback_text}]()"
        else:
            if is_base64_image(image_src):
                return f"![{fallback_text}]()"
            else:
                return f"![{fallback_text}]({image_src})"

    success_count = 0
    images: dict[str, bytes] = {}
    warnings: list[str] = []
    images_descr_costs: float = 0
    image_pattern = re.compile(r'(!\[(.*?)\]\((\S+?)(?:\s+"(.*?)")?\))')
    try:
        md_images = image_pattern.findall(md_text)
        print(f"(doc2poma) Processing {len(md_images)} images…")
        output_md = md_text
        for image_number, (entire_image, alt_text, scr, optional_title) in enumerate(md_images, start=1):
            print(f"(doc2poma) Processing image {image_number}…")
            fallback_text = get_fallback_text_in_md_format(alt_text, optional_title, scr)

            if not scr:
                warnings.append(f"Image {image_number} is missing source.")
                output_md = output_md.replace(entire_image, fallback_text, 1)
                continue

            filetype = None
            image_data = None
            try:
                if is_base64_image(scr):
                    filetype, image_data = extract_type_and_data_from_image_src(scr)
                elif is_wikipedia_image_url(scr):
                    print(f"(doc2poma) Detected Wikipedia image URL: {scr}")
                    wiki_image_url = get_wikipedia_image_url(scr)
                    print(f"(doc2poma) Using Wikipedia image URL: {wiki_image_url}")
                    filetype, image_data = download_type_and_data_from_url(wiki_image_url)
                elif is_relative_url(scr):
                    if scr.startswith("//"):
                        protocol = "https:" if not base_url else base_url.split("://")[0] + ":"
                        src_full_url = protocol + scr
                        print(f"(doc2poma) Processing protocol-relative URL: {scr} -> {src_full_url}")
                        filetype, image_data = download_type_and_data_from_url(src_full_url)
                    else:
                        src_full_url = (base_url or "") + scr
                        filetype, image_data = download_type_and_data_from_url(src_full_url)
                else:
                    filetype, image_data = download_type_and_data_from_url(scr)
            except Exception as exception:
                print(f"(doc2poma) WARNING: Failed to load image {image_number} from {scr}: {exception}")
                warnings.append(f"Image {image_number} failed to load: {exception}")
                output_md = output_md.replace(entire_image, fallback_text, 1)
                continue

            if not image_data:
                warnings.append(f"Image {image_number} failed to load.")
                output_md = output_md.replace(entire_image, fallback_text, 1)
                continue
            if not filetype:
                warnings.append(f"Image {image_number} failed to detect filetype.")
                output_md = output_md.replace(entire_image, fallback_text, 1)
                continue

            image_name = f"image_{image_number:05}.{filetype}"
            images[image_name] = image_data

            if filetype in IMAGE_TYPES_NOT_SUPPORTED_BY_AI_CLIENTS and filetype not in SUPPORTED_IMAGE_TYPES:
                if filetype != "svg":
                    try:
                        image_data = _convert_image_to_png(image_data, filetype)
                        filetype = "png"
                    except Exception as exception:
                        warnings.append(
                            f"Image {image_number} '{filetype}' to 'png' conversion failed: {exception}."
                        )
                        output_md = output_md.replace(entire_image, fallback_text, 1)
                        continue
                else:
                    try:
                        image_data = _convert_svg_image_to_png_best_effort(image_data, image_number, debug)
                        filetype = "png"
                    except Exception as exception:
                        if debug:
                            print(f"(doc2poma) DEBUG: SVG conversion failed: {exception}")
                        if "binary" in str(exception) or "valid SVG" in str(exception):
                            warnings.append(f"Image {image_number} is not a valid SVG file: {exception}")
                        else:
                            warnings.append(
                                f"Image {image_number} 'svg' to 'png' conversion failed: {exception}"
                            )
                        output_md = output_md.replace(entire_image, fallback_text, 1)
                        continue

            if filetype not in SUPPORTED_IMAGE_TYPES:
                warnings.append(
                    f"Image {image_number} format not supported: '{filetype}' (supported: {SUPPORTED_IMAGE_TYPES})"
                )
                output_md = output_md.replace(entire_image, fallback_text, 1)
                continue

            original_size_bytes = _get_size_of_image_data(image_data, filetype)
            if original_size_bytes < 1024:
                warnings.append(f"Image {image_number} is smaller than 1 kB; no description generated.")
                output_md = output_md.replace(entire_image, fallback_text, 1)
                continue

            try:
                description_with_delimiters, img_descr_costs = _convert_image_to_markdown(
                    image_data,
                    filetype,
                    image_number,
                    conversion_params,
                    image_max_size_bytes,
                )
                output_md = output_md.replace(entire_image, f"\n\n{description_with_delimiters}\n\n", 1)
                success_count += 1
                images_descr_costs += img_descr_costs
            except APIRequestTransientError as exception:
                warnings.append(f"Image {image_number} failed after retries: {exception}")
                output_md = output_md.replace(entire_image, fallback_text, 1)
            except Exception as exception:
                warnings.append(f"Image {image_number} failed due to unexpected error: {exception}")
                output_md = output_md.replace(entire_image, fallback_text, 1)

        if warnings:
            warnings_dump = json.dumps(warnings, indent=4)
            print(
                f"(doc2poma) WARNING: {len(warnings)}/{len(md_images)} images failed to process: {warnings_dump}"
            )
        print(f"(doc2poma) Replaced {success_count}/{len(md_images)} images with descriptions.")
    except Exception as exception:
        raise Exception(f"Replacing and saving images in Markdown failed: {exception}")
    return output_md, images, images_descr_costs


def _convert_image_to_png(image_data: bytes, filetype: str) -> bytes:
    """
    Converts image data to PNG format.

    Args:
        image_data (bytes): The image data to be converted.
        filetype (str): The type of the image file.

    Returns:
        bytes: The converted image data in PNG format.
    """
    target_format: str = "PNG"
    img = Image.open(io.BytesIO(image_data))
    if filetype.lower() == "gif":
        try:
            frame = next(ImageSequence.Iterator(img))
            img = frame.copy()
        except StopIteration:
            pass
    elif filetype.lower() in ("tiff", "tif"):
        try:
            img.seek(0)
        except (AttributeError, EOFError):
            pass
    rgb = img.convert("RGB")
    out = io.BytesIO()
    save_kwargs = {}
    rgb.save(out, format=target_format.upper(), **save_kwargs)
    return out.getvalue()


def _convert_svg_image_to_png_best_effort(image_data: bytes, image_number: int, debug: bool = False) -> bytes:
    """
    Tries different methods to convert an SVG file to a PNG file.

    Args:
        image_data (bytes): The SVG image data to be converted.
        image_number (int): The number of the image being processed.

    Returns:
        bytes: The PNG image data after conversion.
    """

    # Validate that the image_data is actually SVG content before attempting conversion
    try:
        start_content = image_data[:100].decode("utf-8", errors="strict")
        is_valid_svg = "<?xml" in start_content.lower() or "<svg" in start_content.lower()
        if not is_valid_svg:
            if debug:
                print(f"(doc2poma) DEBUG: Image data doesn't appear to be valid SVG: {start_content[:50]}...")
            raise Exception("Input data doesn't appear to be valid SVG content")
    except UnicodeDecodeError:
        if debug:
            print(f"(doc2poma) DEBUG: Image data is not valid UTF-8 text, cannot be SVG")
        raise Exception("Input data is binary, not text-based SVG content")

    temp_dir = tempfile.mkdtemp(prefix="poma_svg_conversion_")
    image_path = os.path.join(temp_dir, f"image_{image_number:05}.svg")
    _file_write(image_path, image_data)

    tried_methods = []
    output_path = os.path.join(temp_dir, f"image_{image_number:05}.png")
    output_image_data = None

    def has_lib(name):
        return ctypes.util.find_library(name) is not None

    if not output_image_data and shutil.which("inkscape"):
        tried_methods.append("* Inkscape: Binary found")
        try:
            subprocess.run(
                [
                    "inkscape",
                    image_path,
                    "--export-type=png",
                    f"--export-filename={output_path}",
                ],
                check=True,
            )
            tried_methods.append("* Inkscape: Success")
            output_image_data = _file_read(output_path)
        except Exception as exception:
            tried_methods.append(f"* Inkscape failed: {type(exception).__name__}: {exception}")
    else:
        tried_methods.append("* Inkscape: Not found")

    if not output_image_data and importlib.util.find_spec("cairosvg"):
        if has_lib("cairo"):
            tried_methods.append("* CairoSVG: Installed and 'libcairo' found")
            try:
                import cairosvg

                cairosvg.svg2png(bytestring=b"<svg/>")
                cairosvg.svg2png(url=image_path, write_to=output_path)
                tried_methods.append("* CairoSVG: Success")
                output_image_data = _file_read(output_path)
            except Exception as exception:
                tried_methods.append(f"* CairoSVG failed: {type(exception).__name__}: {exception}")
        else:
            tried_methods.append("* CairoSVG: Installed but 'libcairo' NOT found")
    else:
        tried_methods.append("* CairoSVG: Not installed")

    if not output_image_data and importlib.util.find_spec("wand"):
        if shutil.which("convert"):
            tried_methods.append("* Wand: Installed and ImageMagick 'convert' found")
            try:
                from wand.image import Image

                with Image(filename=image_path) as img:
                    img.format = "png"
                    img.save(filename=output_path)
                tried_methods.append("* Wand: Success")
                output_image_data = _file_read(output_path)
            except Exception as exception:
                tried_methods.append(f"* Wand failed: {type(exception).__name__}: {exception}")
        else:
            tried_methods.append("* Wand: Installed but 'convert' NOT found")
    else:
        tried_methods.append("* Wand: Not installed")

    if not output_image_data and importlib.util.find_spec("svglib") and importlib.util.find_spec("reportlab"):
        tried_methods.append("* svglib+reportlab: Installed")
        try:
            from svglib.svglib import svg2rlg
            from reportlab.graphics import renderPM

            drawing = svg2rlg(image_path)
            renderPM.drawToFile(drawing, output_path, fmt="PNG")
            tried_methods.append("* svglib+reportlab: Success")
            output_image_data = _file_read(output_path)
        except Exception as exception:
            tried_methods.append(f"* svglib+reportlab failed: {type(exception).__name__}: {exception}")
    else:
        tried_methods.append("* svglib+reportlab: Not installed")

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    if not output_image_data:
        tried_methods_dump = json.dumps(tried_methods, indent=4)
        if debug is True:
            print(f"(doc2poma) DEBUG: Tried methods converting svg-image: {tried_methods_dump})")
        raise Exception("No suitable converter found.")
    else:
        return output_image_data


########################
# Image Helper Methods #
########################


def _get_size_of_image_data(image_data: bytes, filetype: str) -> int:
    """
    Get the size of an image from its data.

    Args:
        image_data (bytes): The image data in bytes.
        filetype (str): The format of the image (e.g. 'png').

    Returns:
        int: The size of the image in bytes, or -1 if an error occurs.
    """
    try:
        image = Image.open(io.BytesIO(image_data))
        image_size_bytes, _ = _get_size_and_buffer_of_image(image, image.format or filetype, quality=100)
        return image_size_bytes
    except Exception as exception:
        print(f"(doc2poma) WARNING: getting image size from data failed: {exception}")
        return -1


def _get_size_and_buffer_of_image(image, image_format: str, quality: int) -> tuple[int, io.BytesIO]:
    """
    Get the size and buffer of an image after saving it in a specified format with the given quality.
    Supported formats include JPG/JPEG, HEIC, HEIF, WEBP, and PNG.

    Args:
        image: The input image (PIL.Image or PIL.ImageFile).
        image_format (str): The format of the image (e.g. 'png').
        quality: The quality of the saved image.

    Returns:
        A tuple containing the size of the image buffer and the image buffer itself.
    """
    format = "JPEG" if image_format.upper() == "JPG" else image_format.upper()
    format_options = {
        "JPEG": lambda quality: {"quality": quality},
        "HEIC": lambda quality: {"quality": quality},
        "HEIF": lambda quality: {"quality": quality},
        "WEBP": lambda quality: {"quality": quality, "lossless": False},
        "PNG": lambda _: {"optimize": True},
    }
    if format in format_options:
        save_kwargs = {"format": format}
        save_kwargs.update(format_options[format](quality))
    else:
        raise Exception(f"Error getting image size and buffer: '{format}' not supported.")
    buffer = io.BytesIO()
    try:
        image.save(buffer, **save_kwargs)
    except Exception as exception:
        raise Exception(f"Error getting image size and buffer: {exception}")
    return buffer.tell(), buffer


def _resize_image(image, image_format: str, max_size_bytes: float) -> tuple[BytesIO, float]:
    """
    Resize the input image to fit within the specified maximum size in bytes while maintaining quality.

    Args:
        image: The input image to be resized (PIL.Image or PIL.ImageFile).
        image_format: The format of the input image.
        max_size_bytes: The maximum size in bytes that the resized image should not exceed.

    Returns:
        A tuple containing the resized image as BytesIO object and the final size of the image in bytes.
    """
    quality = 99
    quality_step = 1
    min_quality = 20
    min_width, min_height = 100, 100
    try:
        width, height = image.size
        image_size, output_buffer = _get_size_and_buffer_of_image(image, image_format, quality)
        iteration = 0
        while image_size > max_size_bytes:
            iteration += 1

            scaling_factor = (max_size_bytes / image_size) ** 0.4
            new_width = max(min_width, int(width * scaling_factor))
            new_height = max(min_height, int(height * scaling_factor))

            # Exit if no improvement is possible
            if (new_width == width and new_height == height) and quality <= min_quality:
                # Cannot reduce further without going below min quality or dimensions.
                break

            # Resize if needed
            if new_width != width or new_height != height:
                image = image.resize((new_width, new_height), Image.LANCZOS)
                width, height = image.size

            # Reduce quality if allowed
            if quality > min_quality:
                quality -= quality_step

            image_size, output_buffer = _get_size_and_buffer_of_image(image, image_format, quality)

        final_size_bytes = image_size
        output_buffer.seek(0)
        return output_buffer, final_size_bytes

    except Exception as exception:
        raise Exception(f"Error resizing image: {exception}")


#######################
# File Helper Methods #
#######################


def _file_read(path: str, encoding=None):
    try:
        mode = "rb" if encoding is None else "r"
        with open(path, mode, encoding=encoding) as file:
            return file.read()
    except Exception as exception:
        raise Exception(f"Error reading file '{path}': {exception}")


def _file_write(path: str, data, encoding=None, create_dir_if_not_exists: bool = True) -> str:
    path = path.rstrip(os.path.sep)
    if create_dir_if_not_exists is True:
        dir_path = os.path.dirname(path)
        _create_directory_if_not_exists(dir_path)
    try:
        mode = "wb" if encoding is None else "w"
        data = data if encoding is None else str(data)
        with open(path, mode, encoding=encoding) as file:
            file.write(data)
    except Exception as exception:
        raise Exception(f"Error writing file '{path}': {exception}")
    return path


def _strip_empty_and_br_lines(md_text: str) -> str:
    lines = md_text.splitlines()
    filtered_lines = []
    for line in lines:
        if not line.strip() and len(filtered_lines) > 0 and filtered_lines[-1].strip() == "":
            continue
        if re.match(r"^\s*<br\s*/?\s*>\s*$", line.strip()):
            continue
        filtered_lines.append(line)
    return "\n".join(filtered_lines)


def _create_directory_if_not_exists(dir_path: str):
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            try:
                os.chmod(dir_path, 0o777)
            except Exception as exception:
                print(
                    f"(doc2poma) WARNING: Making directory '{dir_path}' public failed with error: {exception}"
                )
    except FileExistsError as _:
        # Ignore Error if directory already exists
        pass
    except Exception as exception:
        print(f"(doc2poma) WARNING: Creating directory for path '{dir_path}' failed with error: {exception}")


#########
# Other #
#########


def _generate_title_from_markdown(md_text: str, params: dict) -> tuple[str, float]:
    """
    Generate a title from the given Markdown text using an llm.

    Args:
        md_text (str): The Markdown text from which to generate the title.
        params (dict): A dictionary containing parameters for the title generation.

    Returns:
        tuple[str, float]: A tuple containing the generated title and the cost in USD of generating the title.
    """

    provider = str(params["conversion_provider"])
    model = str(params["conversion_model"])
    text_tokens = _num_tokens_from_text(md_text, model)
    if text_tokens > 8000:
        md_text = md_text[:8000]
    prompt = (
        "The text below is in Markdown format (with optional inline HTML tables)."
        "Ignore the formatting characters and focus on the content to create a title that summarizes the main idea."
        "Format the title as a single line of text; do not include any formatting characters."
        "The language of your answer should be the same as the language of the input."
    )
    messages = [
        {"role": "system", "content": f"{prompt}"},
        {"role": "user", "content": f"{md_text}"},
    ]
    litellm.drop_params = True
    response = litellm.completion(
        model=f"{provider}/{model}",
        messages=messages,
        timeout=float(params.get("timeout", 60 * 2)),
        temperature=0.0,
        max_completion_tokens=50,
    )
    indented_content = response.choices[0].message.content
    cost_usd = response._hidden_params.get("response_cost", 0.0)
    if not cost_usd or not isinstance(cost_usd, (int, float)):
        print(
            f"(doc2poma) WARNING: No cost information for generating title - total cost estimated will be inaccurate."
        )
    return indented_content, cost_usd


def _num_tokens_from_text(text: str, model_name: str, buffer: float = 0.2) -> int:
    if text is None or not isinstance(text, str):
        print("(poma-chunker) WARNING: Invalid input for num_tokens_from_text.")
        return 0
    if text == "":
        return 0
    try:
        encoding = _tiktoken_encoding_for_model(model_name)
        return len(encoding.encode(text))
    except Exception:
        # If tiktoken failed, use a conservative character-based estimate
        # and add a small fixed buffer (5%) plus any requested buffer
        char_count = len(text)
        char_based_estimate = char_count / 2.5
        token_count = int(char_based_estimate)
        return int(token_count * (1 + buffer + 0.05)) + 5


def _tiktoken_encoding_for_model(model_name: str) -> tiktoken.Encoding:

    def is_openai_model(model_name: str) -> bool:
        """
        Returns True if the model name refers to an OpenAI model.
        """
        return (
            model_name.startswith("gpt-")
            or model_name.startswith("text-davinci")
            or model_name.startswith("ft:")  # OpenAI fine-tunes
        )

    def get_openai_base_model_name(model_name: str) -> str:
        """
        Extract base model name from OpenAI fine-tuned model name.
        """
        match = re.match(r"^ft:([^:]+)", model_name)
        return match.group(1) if match else model_name

    encoding = None
    if is_openai_model(model_name):
        try:
            base_model_name = get_openai_base_model_name(model_name)
            encoding = tiktoken.encoding_for_model(base_model_name)
        except Exception:
            try:
                encoding = tiktoken.get_encoding("o200k_base")  # gpt-4o, gpt-4o-mini
            except Exception:
                pass
    if encoding is None:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        except Exception:
            pass
    if encoding is None:
        raise Exception(f"Error: No encoding found for model '{model_name}'")
    return encoding
