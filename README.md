## Version 2 is here, ladies and gents!
#### And mere hours after version 1 was released, here's version 2: shallow_langchain_report_with_audio !!
Version 1 file: shallow_langchain_report.py (Kept on repo for usability cases)

Version 2 file: shallow_langchain_report_with_audio.py (Newest and bestest!)

shallow_langchain_report_with_audio combines the first version with implementations of code to support audio files. Thus, you can now run version 2 to submit an audio file (m4a, etc), and it will create a .txt file entitled, "audio_generated_text.txt" in the same directory. From there, it will read that .txt file and produce a summarized report on the content! Is this not the coolest thing ever?

It is still named "shallow", because the backend is not yet implemented -- which results in your output often being cut (the context window is too big for the AI output). This will be solved in Version 2, with a Pinecone backend to help provide longer sessions. Stay tuned for Version 3!


## Usage Instructions
Download the .py files by cloning the repo, or however you may desire.
* To use version 1 (just text-to-report), run the '''shallow_langchain_report.py''' and run it somewhere with a console window (IDE, command line, etc).
* To use version 2 (audio-to-text-to-report), run the '''shallow_langchain_report_with_audio.py''' and run it somewhere with a console window (IDE, command line, etc).

The only other usage requirement is providing it an audio file as user input. We have tested it on .m4a files, which are files from "Voice Memos" application on iPhones. Further filetype testing and support to come later on.


_This tool is a part of Blueprint's AI tool development initiative. Reach out to them! - <a href="https://www.linkedin.com/company/blueprint-servicedesign/" target="_blank">LinkedIn</a>_
