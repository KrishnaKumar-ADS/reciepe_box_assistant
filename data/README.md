# Dataset folder

Place the three original Recipe Box JSON exports here (official archive filenames):

- `recipes_raw_nosource_fn.json`
- `recipes_raw_nosource_epi.json`
- `recipes_raw_nosource_ar.json`

For compatibility, manual uploads with the older filenames omitting `nosource_` are also accepted. The app can download the archive from the official dataset page using the sidebar button, or you can run `python scripts/download_dataset.py` from the project root. The complete third-party dataset is intentionally not bundled in this source ZIP.

Dataset page: <https://eightportions.com/datasets/Recipes/>
Archive URL used by the downloader: <https://eightportions.com/recipes_raw.zip>
