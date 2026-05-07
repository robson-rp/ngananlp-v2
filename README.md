---
title: ElevenAPI quickstart
subtitle: Learn how to make your first ElevenLabs API request.
---

By the end of this guide you will have a working script that sends a text string to the ElevenLabs API and plays the returned audio through your speakers. You will learn how to authenticate with an API key, install the SDK, and make your first text-to-speech request.

For guides covering other capabilities — streaming, voice cloning, speech-to-text — see the [Tutorials](/docs/eleven-api/guides/cookbooks) section.

<Tip>
  Use the [ElevenLabs text-to-speech skill](https://github.com/elevenlabs/skills/tree/main/text-to-speech) to generate speech from your AI coding assistant:

```bash
npx skills add elevenlabs/skills --skill text-to-speech
```

</Tip>

## Using the Text to Speech API

<Steps>
    <Step title="Create an API key">
      [Create an API key in the dashboard here](https://elevenlabs.io/app/settings/api-keys), which you’ll use to securely [access the API](/docs/api-reference/authentication).
      
      Store the key as a managed secret and pass it to the SDKs either as a environment variable via an `.env` file, or directly in your app’s configuration depending on your preference.
      
      ```js title=".env"
      ELEVENLABS_API_KEY=<your_api_key_here>
      ```
      
    </Step>
    <Step title="Install the SDK">
      We'll also use the `dotenv` library to load our API key from an environment variable.
      
      <CodeBlocks>
          ```python
          pip install elevenlabs
          pip install python-dotenv
          ```
      
          ```typescript
          npm install @elevenlabs/elevenlabs-js
          npm install dotenv
          ```
      
      </CodeBlocks>
      

      <Note>
        To play the audio through your speakers, you may be prompted to install [MPV](https://mpv.io/)
      and/or [ffmpeg](https://ffmpeg.org/).
      </Note>
    </Step>
    <Step title="Make your first request">
      Create a new file named `example.py` or `example.mts`, depending on your language of choice and add the following code:
       <CodeBlocks>
       ```python
       from dotenv import load_dotenv
       from elevenlabs.client import ElevenLabs
       from elevenlabs.play import play
       import os
       
       load_dotenv()
       
       elevenlabs = ElevenLabs(
         api_key=os.getenv("ELEVENLABS_API_KEY"),
       )
       
       audio = elevenlabs.text_to_speech.convert(
           text="The first move is what sets everything in motion.",
           voice_id="JBFqnCBsd6RMkjVDRZzb",  # "George" - browse voices at elevenlabs.io/app/voice-library
           model_id="eleven_v3",
           output_format="mp3_44100_128",
       )
       
       play(audio)
       
       ```
       
       ```typescript
       import { ElevenLabsClient, play } from '@elevenlabs/elevenlabs-js';
       import 'dotenv/config';
       
       const elevenlabs = new ElevenLabsClient();
       const audio = await elevenlabs.textToSpeech.convert(
         'JBFqnCBsd6RMkjVDRZzb', // "George" - browse voices at elevenlabs.io/app/voice-library
         {
           text: 'The first move is what sets everything in motion.',
           modelId: 'eleven_v3',
           outputFormat: 'mp3_44100_128',
         }
       );
       
       await play(audio);
       
       ```
       
       </CodeBlocks>
    </Step>
    <Step title="Run the code">
        <CodeBlocks>
            ```python
            python example.py
            ```

            ```typescript
            npx tsx example.mts
            ```
        </CodeBlocks>

        You should hear the audio play through your speakers.
    </Step>

</Steps>

## Next steps

<CardGroup cols={3}>
  <Card
    title="Stream audio"
    icon="file:98838ac6-bdff-429b-ad27-edefb93ec1f8"
    href="/docs/eleven-api/guides/how-to/text-to-speech/streaming"
  >
    Reduce latency by streaming audio as it generates rather than waiting for the complete file
  </Card>
  <Card
    title="Browse voices"
    icon="file:9d711ebb-cba8-4d3e-a0dc-6e7672194b78"
    href="https://elevenlabs.io/app/voice-library"
  >
    Explore 10,000+ voices and swap the example voice ID for one that fits your use case
  </Card>
  <Card
    title="Clone a voice"
    icon="file:8c53739c-dd1d-454f-966b-6cb3f335381b"
    href="/docs/eleven-api/guides/how-to/voices/instant-voice-cloning"
  >
    Create a custom voice from a short audio recording
  </Card>
</CardGroup>