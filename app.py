import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import textwrap
import torch  # Ensure PyTorch is imported
import time

def extract_video_id(url):
    """Extract video ID from YouTube URL."""
    from urllib.parse import urlparse, parse_qs

    parsed_url = urlparse(url)

    if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        return parse_qs(parsed_url.query).get('v', [None])[0]
    elif parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]

    return None

def summarize_video(video_url):
    try:
        # Extract video ID from URL
        video_id = extract_video_id(video_url)

        if not video_id:
            print("Invalid YouTube URL")
            return

        # Fetch transcript using YouTubeTranscriptApi
        transcript = YouTubeTranscriptApi.get_transcript(video_id)

        # Save transcript to a txt file
        with open(f"transcript/{video_id}.txt", "w") as file:
            for entry in transcript:
                file.write(entry['text'] + "\n")

        # Combine transcript into a single text
        full_text = " ".join([entry['text'] for entry in transcript])

        # Summarize the text
        summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            device=0 if torch.cuda.is_available() else -1
        )

        # Split text into chunks to handle long input sequences
        max_input_length = 1024
        chunks = []
        for i in range(0, len(full_text), max_input_length):
            end = i + max_input_length
            chunks.append(full_text[i:end])

        summaries = []
        for chunk in chunks:
            summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
            summaries.append(summary[0]['summary_text'])

        final_summary = " ".join(summaries)

        # Save summarizer response to a txt file
        with open(f"summary/{video_id}.txt", "w") as file:
            file.write(final_summary)

        print("Summary:")
        print(final_summary)
        return final_summary

    except Exception as e:
        print(f"An error occurred: {str(e)}")

def generate_video_script(summary_text):
    """Generate a short video script from the summary."""
    try:
        generator = pipeline(
            "text-generation",
            model="gpt2",
            device=0 if torch.cuda.is_available() else -1
        )

        prompt = textwrap.dedent(
            f"""
            Create a concise and engaging script for a short video based on the following summary. The script should include a brief intro, key points, and an outro.\n\nSummary:\n{summary_text}\n\nScript:
            """
        ).strip()

        generated = generator(prompt, max_length=200, num_return_sequences=1)
        script = generated[0]["generated_text"][len(prompt):].strip()
        return script
    except Exception as e:
        print(f"An error occurred while generating the script: {str(e)}")
        return ""

# Streamlit app
st.title("YouTube Video Summarizer")

tab1, tab2 = st.tabs(["Summarize Video", "Generate Video Script"])

with tab1:
    url = st.text_input("Enter YouTube URL:", key="summarize_url")
    if st.button("Summarize", key="summarize_button"):
        with st.spinner('Summarizing...'):
            start_time = time.time()
            summary = summarize_video(url)
            end_time = time.time()
            time_taken = end_time - start_time
            st.header("Summary:")
            st.markdown(summary)
            st.success(f"Time taken: {time_taken:.2f} seconds")

with tab2:
    url_script = st.text_input("Enter YouTube URL:", key="script_url")
    if st.button("Create Script", key="script_button"):
        with st.spinner('Generating script...'):
            start_time = time.time()
            summary = summarize_video(url_script)
            script = generate_video_script(summary)
            end_time = time.time()
            time_taken = end_time - start_time
            st.header("Summary:")
            st.markdown(summary)
            st.header("Video Script:")
            st.markdown(script)
            st.success(f"Time taken: {time_taken:.2f} seconds")