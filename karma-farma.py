import os
import asyncio
import aiohttp
import asyncpraw
import logging
import argparse
from urllib.parse import urlparse
from dotenv import load_dotenv
from pathlib import Path
from PIL import Image  # Pillow is used by imagehash
import imagehash
import concurrent.futures
from collections import defaultdict

# --- Configuration ---

# Load environment variables from .env file
# Ensure you have: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT, IMGUR_CLIENT_ID
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Allowed media file extensions for download
DOWNLOAD_EXTENSIONS = {
    ".jpg",
    ".png",
    ".gif",
    ".jpeg",
    ".mp4",
    ".gifv",
    ".webm",
    ".webp",
}
# Supported image extensions for deduplication (adjust as needed)
DEDUP_SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".tiff",
    ".webm",
    ".webp",
}

# Default concurrent downloads/workers
DEFAULT_MAX_WORKERS = 10

# --- Deduplication Functions (Integrated) ---


def compute_hash(
    image_path: Path, hash_size: int
) -> tuple[Path, imagehash.ImageHash | None, str | None]:
    """
    Computes the perceptual hash for a single image file.

    Args:
        image_path: Path object for the image file.
        hash_size: Size of the hash to compute (e.g., 8, 16).

    Returns:
        A tuple containing:
        - The original image_path.
        - The computed imagehash.ImageHash object, or None if hashing failed.
        - An error message string if an error occurred, otherwise None.
    """
    try:
        # Use Pillow to open the image, as required by imagehash
        img = Image.open(image_path)
        # Compute perceptual hash (pHash is generally good for photos)
        hash_val = imagehash.phash(img, hash_size=hash_size)
        return image_path, hash_val, None
    except Exception as e:
        # Log error for specific file, but allow the process to continue
        error_msg = f"Error processing {image_path} for hashing: {e}"
        # Use a distinct logger or message prefix if needed
        logging.warning(f"[Deduplication] {error_msg}")
        return image_path, None, error_msg


def find_and_remove_duplicates(
    directory: str,
    hash_size: int = 8,
    dry_run: bool = False,
    max_workers: int | None = None,
):
    """
    Finds and removes visually duplicate images in a directory and its subdirectories
    using perceptual hashing and parallel processing. (Called after download)

    Args:
        directory: The root directory to scan for images (e.g., ./downloads/subreddit_name).
        hash_size: The size of the perceptual hash.
        dry_run: If True, only log files that would be deleted.
        max_workers: Maximum number of worker processes for hashing.
    """
    root_dir = Path(directory)
    if not root_dir.is_dir():
        logging.error(f"[Deduplication] Error: Directory not found: {directory}")
        return

    logging.info(f"[Deduplication] Starting scan in: {root_dir}")
    # Scan recursively within the target directory
    image_paths = [
        p
        for p in root_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in DEDUP_SUPPORTED_EXTENSIONS
    ]

    if not image_paths:
        logging.info("[Deduplication] No image files found for deduplication.")
        return

    logging.info(
        f"[Deduplication] Found {len(image_paths)} image files. Computing hashes..."
    )

    # --- Hashing Phase (Parallel) ---
    hashes = defaultdict(list)  # {hash_value: [path1, path2, ...]}
    error_files = []

    # Use ProcessPoolExecutor for CPU-bound hashing task
    # Note: This runs synchronously after the async download part.
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(compute_hash, path, hash_size) for path in image_paths
        ]
        processed_count = 0
        for future in concurrent.futures.as_completed(futures):
            processed_count += 1
            try:
                path, hash_val, error_msg = future.result()
                if error_msg:
                    error_files.append((path, error_msg))
                elif hash_val is not None:
                    hashes[hash_val].append(path)
                if processed_count % 50 == 0 or processed_count == len(image_paths):
                    logging.info(
                        f"[Deduplication] Hashed {processed_count}/{len(image_paths)} images..."
                    )
            except Exception as e:
                logging.error(
                    f"[Deduplication] Unexpected error retrieving hash result: {e}"
                )

    logging.info("[Deduplication] Hashing complete.")
    if error_files:
        logging.warning(
            f"[Deduplication] Encountered errors processing {len(error_files)} files during hashing."
        )
        # Optionally log details here

    # --- Duplicate Identification and Deletion Phase ---
    duplicates_found = 0
    files_to_delete = []

    logging.info("[Deduplication] Identifying duplicates based on hashes...")
    for hash_val, paths in hashes.items():
        if len(paths) > 1:
            paths.sort()  # Sort for consistency
            files_to_keep = paths[0]
            files_to_remove = paths[1:]
            duplicates_found += len(files_to_remove)
            files_to_delete.extend(files_to_remove)
            logging.debug(
                f"[Deduplication] Hash {hash_val}: Keeping '{files_to_keep}', Duplicates: {[str(p) for p in files_to_remove]}"
            )

    if not files_to_delete:
        logging.info("[Deduplication] No duplicate images found.")
        return

    logging.info(
        f"[Deduplication] Found {duplicates_found} duplicate image file(s) to remove."
    )

    if dry_run:
        logging.info("[Deduplication Dry Run] Would delete the following files:")
        for f_path in files_to_delete:
            print(f"  - {f_path}")  # Print for dry run clarity
    else:
        logging.info("[Deduplication] Deleting duplicate files...")
        deleted_count = 0
        failed_deletions = []
        for f_path in files_to_delete:
            try:
                f_path.unlink()
                logging.info(f"[Deduplication] Deleted: {f_path}")
                deleted_count += 1
            except OSError as e:
                logging.error(f"[Deduplication] Failed to delete {f_path}: {e}")
                failed_deletions.append(f_path)
        logging.info(f"[Deduplication] Successfully deleted {deleted_count} files.")
        if failed_deletions:
            logging.warning(
                f"[Deduplication] Failed to delete {len(failed_deletions)} files."
            )

    logging.info("[Deduplication] Process finished.")


# --- Reddit Downloader Functions ---


async def get_redgifs_token(session: aiohttp.ClientSession) -> str | None:
    """Asynchronously fetches a temporary Redgifs API token."""
    url = "https://api.redgifs.com/v2/auth/temporary"
    headers = {
        "Accept": "application/json",
        "User-Agent": os.getenv("REDDIT_USER_AGENT", "python:reddit-downloader:v3.0"),
    }
    try:
        async with session.get(url, headers=headers) as response:
            response.raise_for_status()
            token_data = await response.json()
            token_string = token_data.get("token")
            if token_string:
                logging.info("Successfully obtained Redgifs token.")
                return token_string
            else:
                logging.error(f"Failed to get Redgifs token: 'token' key not found.")
                return None
    except Exception as e:
        logging.error(f"Error fetching Redgifs token: {e}")
        return None


async def download_media(
    session: aiohttp.ClientSession,
    url: str,
    filepath: str,
    semaphore: asyncio.Semaphore,
    redgifs_token: str | None = None,
    imgur_client_id: str | None = None,
):
    """Downloads a single media file asynchronously."""
    async with semaphore:
        try:
            if os.path.exists(filepath):
                logging.info(f"File already exists, skipping download: {filepath}")
                return

            headers = {}
            if "redgifs.com" in url and redgifs_token:
                headers["Authorization"] = f"Bearer {redgifs_token}"
            elif "imgur.com" in url and imgur_client_id:
                headers["Authorization"] = f"Client-ID {imgur_client_id}"

            logging.info(f"Attempting download: {url} -> {filepath}")
            async with session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    with open(filepath, "wb") as fd:
                        while True:
                            chunk = await resp.content.read(1024 * 8)
                            if not chunk:
                                break
                            fd.write(chunk)
                    logging.info(f"Successfully downloaded: {filepath}")
                else:
                    logging.error(
                        f"Failed download {url}: HTTP {resp.status} {resp.reason}"
                    )
        except Exception as e:
            logging.error(f"Error downloading {url}: {e}")


async def fetch_reddit_submissions(
    reddit: asyncpraw.Reddit,
    source_type: str,
    source_name: str,
    sort_order: str,
    limit: int,
    time_filter: str | None = None,
) -> list[asyncpraw.models.Submission]:
    """Fetches submissions from a specified Reddit source."""
    submissions = []
    try:
        logging.info(
            f"Fetching {limit} posts from {source_type} '{source_name}' sorted by '{sort_order}' ({time_filter or 'N/A'})"
        )
        if source_type == "subreddit":
            source = await reddit.subreddit(source_name)
            await source.load()
            if sort_order == "hot":
                stream = source.hot(limit=limit)
            elif sort_order == "new":
                stream = source.new(limit=limit)
            else:
                stream = source.top(
                    limit=limit, time_filter=time_filter or "all"
                )  # Default 'all' for top
        elif source_type == "user":
            source = await reddit.redditor(source_name)
            await source.load()
            if sort_order == "top":
                stream = source.submissions.top(
                    limit=limit, time_filter=time_filter or "all"
                )
            else:
                stream = source.submissions.new(
                    limit=limit
                )  # Default to new for user if not top
        else:
            logging.error(f"Invalid source type: {source_type}")
            return []

        async for submission in stream:
            submissions.append(submission)
        logging.info(f"Fetched {len(submissions)} submissions.")
        return submissions

    except Exception as e:
        logging.error(f"Error fetching data for {source_type} '{source_name}': {e}")
        return []


def delete_empty_folders(base_dir: str):
    """Recursively deletes empty folders within the specified base directory."""
    deleted_count = 0
    if not os.path.isdir(base_dir):
        logging.warning(f"Base directory for cleanup not found: {base_dir}")
        return

    logging.info(f"Cleaning empty folders in {base_dir}...")
    for root, dirs, files in os.walk(base_dir, topdown=False):
        is_empty = not dirs and not files
        # Extra check to ensure we don't delete the base_dir itself if it happens to be empty
        if is_empty and Path(root) != Path(base_dir):
            try:
                os.rmdir(root)
                logging.info(f"Deleted empty folder: {root}")
                deleted_count += 1
            except OSError as e:
                logging.error(f"Error deleting folder {root}: {e}")
    if deleted_count > 0:
        logging.info(
            f"Finished cleaning empty folders. Deleted {deleted_count} directories."
        )
    else:
        logging.info("No empty folders found to delete.")


# --- Main Execution ---


async def main():
    """Main function to parse arguments, fetch data, download media, and deduplicate."""
    parser = argparse.ArgumentParser(
        description="Download media from Reddit and optionally remove visual duplicates."
    )
    # Downloader Args
    parser.add_argument(
        "source_type",
        choices=["subreddit", "user"],
        help="Specify 'subreddit' or 'user'.",
    )
    parser.add_argument("source_name", help="Name of the subreddit or user.")
    parser.add_argument(
        "-s",
        "--sort",
        choices=["hot", "new", "top"],
        default="hot",
        help="Sort order for posts.",
    )
    parser.add_argument(
        "-t",
        "--time",
        choices=["all", "year", "month", "week", "day", "hour"],
        default="all",
        help="Time filter for 'top' sort.",
    )
    parser.add_argument(
        "-l", "--limit", type=int, default=25, help="Number of posts to fetch."
    )
    parser.add_argument(
        "-d",
        "--dir",
        default="reddit_downloads",
        help="Base directory to save downloaded media.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Maximum number of concurrent downloads/hash workers.",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Disable deleting empty folders after download.",
    )

    # Deduplication Args
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Enable visual duplicate removal after download.",
    )
    parser.add_argument(
        "--dedup-hash-size",
        type=int,
        default=8,
        help="Perceptual hash size for deduplication (default: 8).",
    )
    parser.add_argument(
        "--dedup-dry-run",
        action="store_true",
        help="Perform a dry run for deduplication (log only).",
    )

    args = parser.parse_args()

    if args.limit <= 0:
        logging.error("Post limit must be a positive integer.")
        return

    # Get environment variables
    # ... (same credential checking as before) ...
    reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
    reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    reddit_user_agent = os.getenv("REDDIT_USER_AGENT")
    imgur_client_id = os.getenv("IMGUR_CLIENT_ID")  # Optional

    if not all([reddit_client_id, reddit_client_secret, reddit_user_agent]):
        logging.error("Missing Reddit API credentials in environment variables.")
        return

    # Initialize shared resources
    semaphore = asyncio.Semaphore(args.max_concurrent)
    download_tasks = []
    processed_urls = set()
    # Define the specific target directory for this run's downloads & potential deduplication
    run_target_dir = os.path.join(args.dir, args.source_name)

    async with aiohttp.ClientSession() as session:
        redgifs_token = await get_redgifs_token(session)
        reddit = asyncpraw.Reddit(
            client_id=reddit_client_id,
            client_secret=reddit_client_secret,
            user_agent=reddit_user_agent,
            requestor_kwargs={"session": session},
        )

        try:
            submissions = await fetch_reddit_submissions(
                reddit,
                args.source_type,
                args.source_name,
                args.sort,
                args.limit,
                args.time if args.sort == "top" else None,
            )

            if not submissions:
                logging.warning("No submissions found or fetched.")
            else:
                for submission in submissions:
                    if (
                        hasattr(submission, "url")
                        and submission.url
                        and any(
                            submission.url.lower().endswith(ext)
                            for ext in DOWNLOAD_EXTENSIONS
                        )
                    ):

                        url = submission.url
                        if url in processed_urls:
                            continue

                        author_name = (
                            str(submission.author)
                            if submission.author
                            else "deleted_user"
                        )
                        path_parts = urlparse(url).path.split("/")
                        filename = (
                            path_parts[-1] if path_parts else f"unknown_{submission.id}"
                        )
                        filename = "".join(
                            c for c in filename if c.isalnum() or c in (".", "_", "-")
                        ).strip()
                        if not filename:
                            filename = f"media_{submission.id}{os.path.splitext(urlparse(url).path)[1]}"

                        # Place files directly under run_target_dir/author_name/
                        download_dir = os.path.join(run_target_dir, author_name)
                        filepath = os.path.join(download_dir, filename)

                        task = download_media(
                            session,
                            url,
                            filepath,
                            semaphore,
                            redgifs_token,
                            imgur_client_id,
                        )
                        download_tasks.append(task)
                        processed_urls.add(url)

                if download_tasks:
                    logging.info(
                        f"Starting download of {len(download_tasks)} media files..."
                    )
                    await asyncio.gather(*download_tasks)
                    logging.info("All download tasks finished.")
                else:
                    logging.info(
                        "No valid media URLs found in the fetched submissions."
                    )

        except Exception as e:
            logging.exception(
                f"An unexpected error occurred during download phase: {e}"
            )
        finally:
            if "reddit" in locals() and reddit:
                await reddit.close()
                logging.info("Reddit instance closed.")

    # --- Post-Download Steps (Synchronous) ---

    # 1. Deduplication (if enabled) - Run BEFORE empty folder cleanup
    if args.deduplicate:
        logging.info("-" * 20)
        logging.info("Starting Deduplication Phase...")
        # Run deduplication on the directory created for this specific source
        find_and_remove_duplicates(
            directory=run_target_dir,  # Pass the specific target dir
            hash_size=args.dedup_hash_size,
            dry_run=args.dedup_dry_run,
            max_workers=args.max_concurrent,  # Reuse concurrency setting
        )
        logging.info("Deduplication Phase Finished.")
        logging.info("-" * 20)

    # 2. Cleanup Empty Folders (if enabled)
    if not args.no_cleanup:
        logging.info("Starting cleanup of empty folders...")
        # Clean up within the specific target directory for this run
        delete_empty_folders(run_target_dir)
    else:
        logging.info("Skipping cleanup of empty folders.")

    logging.info("Script finished.")


if __name__ == "__main__":
    asyncio.run(main())
