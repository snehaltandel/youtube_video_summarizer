# YouTube Video Summarizer

A Streamlit app that pulls a YouTube video's transcript and summarizes it
using a Hugging Face transformers pipeline, with a second tab that turns the
summary into a short video script.

## Features

- Extracts a video's transcript via `youtube-transcript-api`
- Summarizes the transcript with a transformers summarization pipeline
- Generates a short-form video script from the summary
- Saves transcripts/summaries to `transcript/` and `summary/`

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Paste a YouTube URL into the "Summarize Video" tab to get a summary, or use
the "Generate Video Script" tab to turn that summary into a script.

## License

MIT — see [LICENSE](LICENSE).
