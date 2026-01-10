# Audio

## Create speech

> post https://localhost:8080/v1/audio/speech

OpenAI's [createSpeech](https://platform.openai.com/docs/api-reference/audio/createSpeech) docs.

### Sample

```shell
curl -X POST "http://localhost:8100/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lucasnewman/f5-tts-mlx",
    "input": "MLX project is awsome",
    "voice": "alloy"
  }' \
  --output ~/Desktop/mlx.wav
```

## Create transcription

> post https://localhost:8080/v1/audio/transcriptions

OpenAI's [createTranscription](https://platform.openai.com/docs/api-reference/audio/createTranscription) docs.

### Sample

<details>
<summary>Default json sample</summary>

```shell
curl -X POST "http://localhost:8100/v1/audio/transcriptions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@mlx_example.wav" \
  -F "model=mlx-community/whisper-large-v3-turbo"
```

```json
{
  "text": " MLX Project is awesome!"
}
```

</details>

---

<details>
<summary>Text sample</summary>

```shell
curl -X POST "http://localhost:8100/audio/transcriptions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@mlx_example.wav" \
  -F "model=mlx-community/whisper-large-v3-turbo" \
  -F "response_format=text"
```

```text
MLX Project is awesome!
```

</details>

---

<details>
<summary>SRT sample</summary>

```shell
curl -X POST "http://localhost:8100/audio/transcriptions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@mlx_example.wav" \
  -F "model=mlx-community/whisper-large-v3-turbo" \
  -F "response_format=srt"
```

```text
1
00:00:00,000 --> 00:00:03,000
MLX Project is awesome!
```

</details>

---

<details>
<summary>Verbose json sample</summary>

```shell
curl -X POST "http://localhost:8100/audio/transcriptions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@mlx_example.wav" \
  -F "model=mlx-community/whisper-large-v3-turbo" \
  -F "response_format=verbose_json"
```

```text
{
	"text": " MLX Project is awesome!",
	"segments": [{
		"id": 0,
		"seek": 0,
		"start": 0.0,
		"end": 3.0,
		"text": " MLX Project is awesome!",
		"tokens": [50364, 21601, 55, 9849, 307, 3476, 0, 50514],
		"temperature": 0.0,
		"avg_logprob": -0.6914801067776151,
		"compression_ratio": 0.7419354838709677,
		"no_speech_prob": 0.04768477380275726
	}],
	"language": "en"
}
```

</details>

---

<details>
<summary>Verbose word json sample</summary>

```shell
curl -X POST "http://localhost:8100/audio/transcriptions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@mlx_example.wav" \
  -F "model=mlx-community/whisper-large-v3-turbo" \
  -F "timestamp_granularities[]=word" \
  -F "response_format=verbose_json"
```

```text
{
	"text": " MLX project is awesome.",
	"segments": [{
		"id": 0,
		"seek": 0,
		"start": 0.0,
		"end": 2.58,
		"text": " MLX project is awesome.",
		"tokens": [50364, 21601, 55, 1716, 307, 3476, 13, 50514],
		"temperature": 0.0,
		"avg_logprob": -0.7067226303948296,
		"compression_ratio": 0.7419354838709677,
		"no_speech_prob": 0.13033607602119446,
		"words": [{
			"word": " MLX",
			"start": 0.0,
			"end": 1.48,
			"probability": 0.80322265625
		}, {
			"word": " project",
			"start": 1.48,
			"end": 1.88,
			"probability": 0.52197265625
		}, {
			"word": " is",
			"start": 1.88,
			"end": 2.12,
			"probability": 0.998046875
		}, {
			"word": " awesome.",
			"start": 2.12,
			"end": 2.58,
			"probability": 0.96533203125
		}]
	}],
	"language": "en"
}
```

</details>

---

<details>

<summary>VTT sample</summary>

```shell
curl -X POST "http://localhost:8100/v1/audio/transcriptions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@mlx_example.wav" \
  -F "model=mlx-community/whisper-large-v3-turbo" \
  -F "response_format=vtt"
```

```text
WEBVTT

00:00.000 --> 00:03.000
MLX Project is awesome!
```

</details>
