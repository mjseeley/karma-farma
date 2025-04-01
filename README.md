# KARMA-FARMA (Reddit Media Downloader & Deduplicator)

This Python script downloads media (images, GIFs, videos) from specified Reddit subreddits or users and includes an option to automatically remove visually duplicate images after downloading.

## Features

- Download media from subreddits or individual user profiles.
- Filter posts by 'hot', 'new', or 'top' (with time filters for 'top').
- Specify the number of posts to check.
- Customize the download directory structure (`base_dir/source_name/author_name/filename.ext`).
- **Optional:** Automatically detect and remove visually duplicate images within the downloaded content for the current run using perceptual hashing.
- Utilizes asynchronous downloading for speed.
- Uses parallel processing for efficient duplicate detection.
- Provides options for dry runs (for deduplication) and cleanup of empty folders.

## Prerequisites

1.  **Python:** Python 3.8 or higher is recommended.
2.  **Libraries:** Install the required libraries using pip:
    ```bash
    pip install aiohttp asyncpraw python-dotenv imagehash Pillow
    ```
3.  **.env File:** Create a file named `.env` in the same directory as the script and add your Reddit API credentials (and optionally Imgur):
    ```dotenv
    REDDIT_CLIENT_ID=your_reddit_client_id
    REDDIT_CLIENT_SECRET=your_reddit_client_secret
    REDDIT_USER_AGENT=YourAppName (by /u/YourUsername)
    IMGUR_CLIENT_ID=your_imgur_client_id # Optional, only needed if downloading directly from Imgur links often
    ```
    - You can obtain Reddit API credentials by creating an app on Reddit's [app preferences page](https://www.reddit.com/prefs/apps). Select "script" as the app type.
    - The `REDDIT_USER_AGENT` should be a unique descriptive string, including your Reddit username is good practice.

## Usage

Run the script from your terminal using `python karma-farma.py` followed by the source type, source name, and any desired options.

```bash
python karma-farma.py <source_type> <source_name> [options]
Examples:Download the top 50 posts from the 'pics' subreddit this week and remove duplicates:python karma-farma.py subreddit pics -s top -t week -l 50 --deduplicate
Download the 100 newest posts from user 'SpecificRedditor' into a custom directory, without cleaning empty folders afterwards:python karma-farma.py user SpecificRedditor -s new -l 100 -d ./my_reddit_saves --no-cleanup
Perform a dry run of downloading and deduplicating the top 20 posts of all time from 'funny':python karma-farma.py subreddit funny -s top -t all -l 20 --deduplicate --dedup-dry-run
Command-Line OptionsRequired Argumentssource_type: Specify the type of source.Choices: subreddit, usersource_name: The name of the subreddit (e.g., pics) or the Reddit username (e.g., SpecificRedditor).Downloader Options-s, --sort: Sort order for fetching posts.Choices: hot, new, topDefault: hot-t, --time: Time filter when using --sort top. Ignored otherwise.Choices: all, year, month, week, day, hourDefault: all-l, --limit: Maximum number of posts to fetch and check for media.Type: integerDefault: 25-d, --dir: Base directory where downloaded media will be saved. Subdirectories for the source name and author will be created inside this directory.Type: stringDefault: reddit_downloadsDeduplication Options--deduplicate: Enable visual duplicate removal using perceptual hashing after downloads are complete. This operates only on the files downloaded in the current run for the specified source.Type: flag (presence enables it)--dedup-hash-size: The sensitivity of the perceptual hash for deduplication. Smaller values are faster but less sensitive; larger values are more sensitive but slower. Powers of 2 (like 8, 16) are common.Type: integerDefault: 8--dedup-dry-run: If --deduplicate is enabled, this flag prevents actual file deletion. Instead, it logs the duplicate files that would have been removed. Useful for testing.Type: flag (presence enables it)General Options--max-concurrent: Maximum number of concurrent operations. This limits both simultaneous downloads and parallel workers used for hashing during deduplication. Adjust based on your network and CPU capabilities.Type: integerDefault: 10--no-cleanup: Disable the automatic deletion of empty folders within the target download directory after the script finishes (including after deduplication).Type: flag (presence enables it)-h, --help: Show the help message listing
```
