package com.example.talkandexecute

import android.app.Application
import android.media.MediaRecorder
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.talkandexecute.classification.AudioClassificationHelper
import com.example.talkandexecute.model.GeneratedAnswer
import com.example.talkandexecute.model.SpeechState
import com.google.gson.Gson
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.logging.HttpLoggingInterceptor
import org.json.JSONArray
import org.json.JSONObject
import org.tensorflow.lite.support.label.Category
import java.io.File
import java.io.IOException
import java.lang.RuntimeException
import java.util.concurrent.TimeUnit

class ChatGPTViewModel(application: Application) : AndroidViewModel(application) {

    var speechState by mutableStateOf(SpeechState())

    private var mediaRecorder: MediaRecorder = MediaRecorder()
    private var isRecording: Boolean = false
    private var numberOfBackgroundLabel = 0
    private val outputFile = File(application.filesDir, "recording.mp3")
    private val audioClassificationListener = object : AudioClassificationListener {
        override fun onResult(results: List<Category>, inferenceTime: Long) {
            Log.v("speech_result", "$results $inferenceTime")
            if (results.isNotEmpty()) {
                if (results[0].index == 7) {
                    numberOfBackgroundLabel = 0
                    startListening()
                } else if (results[0].index == 0) {
                    if (isRecording) {
                        // Log.v("speech_number", "$numberOfBackgroundLabel")
                        numberOfBackgroundLabel++
                        if (numberOfBackgroundLabel > 10) {
                            numberOfBackgroundLabel = 0
                            stopListening()
                        }
                    }
                }
            } else {
                numberOfBackgroundLabel++
            }
        }

        override fun onError(error: String) {
            Log.v("speech_result", error)
        }
    }
    private val audioClassificationHelper = AudioClassificationHelper(context = application, listener = audioClassificationListener)

    init {
        audioClassificationHelper.initClassifier()
    }

    fun startListening() {
        // Log.v("speech_start", "start")
        if (!isRecording) {
            isRecording = true
            numberOfBackgroundLabel = 0
            // Log.v("speech_start", "true")
            try {
                mediaRecorder.apply {
                    // Initialization.
                    setAudioSource(MediaRecorder.AudioSource.MIC)
                    setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
                    setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
                    // path to recording, if you want to hear it.
                    setOutputFile(outputFile.absolutePath)
                }
                mediaRecorder.prepare()
                mediaRecorder.start()
            } catch (e: IllegalStateException) {
                Log.e(TAG, e.toString())
                // Handle the exception -> MediaRecorder is not in the initialized state
                mediaRecorder.reset()
                isRecording = false
                numberOfBackgroundLabel = 0
            } catch (e: IOException) {
                Log.e(TAG, e.toString())
                // Handle the exception -> failed to prepare MediaRecorder
                mediaRecorder.reset()
                isRecording = false
                numberOfBackgroundLabel = 0
            }
        }
    }

    fun stopListening() {
        // Log.v("speech_stop", "stop")
        if (isRecording) {
            // Log.v("speech_stop", "true")
            try {
                mediaRecorder.stop()
                mediaRecorder.reset()
                isRecording = false

                viewModelScope.launch(Dispatchers.Default) {
                    val transcribedText = transcribeAudio(outputFile)
                    speechState = try {
                        speechState.copy(speechResult = transcribedText)
                    } catch (e: IOException) {
                        // There was an error
                        speechState.copy(speechResult = "API Error: ${e.message}")
                    }
                    speechState = try {
                        val returnedText = createChatCompletion(transcribedText)
                        mediaRecorder.reset()
                        isRecording = false
                        numberOfBackgroundLabel = 0
                        speechState.copy(palmResult = returnedText)
                    } catch (e: IOException) {
                        // There was an error
                        mediaRecorder.reset()
                        isRecording = false
                        numberOfBackgroundLabel = 0
                        speechState.copy(palmResult = "API Error: ${e.message}")
                    }
                }
            } catch (e: RuntimeException) {
                Log.e(TAG, e.toString())
                // Handle the exception -> state machine is not in a valid state
                mediaRecorder.reset()
                isRecording = false
                numberOfBackgroundLabel = 0
            } catch (e: IllegalStateException) {
                Log.e(TAG, e.toString())
                mediaRecorder.reset()
                isRecording = false
                numberOfBackgroundLabel = 0
            }
        }
    }

    private var loggingInterceptor = HttpLoggingInterceptor().setLevel(HttpLoggingInterceptor.Level.BODY)
    private val client = OkHttpClient.Builder()
        .addInterceptor(loggingInterceptor)
        .connectTimeout(30, TimeUnit.SECONDS) // Set connection timeout to 30 seconds
        .readTimeout(30, TimeUnit.SECONDS)    // Set read timeout to 30 seconds
        .build()

    private fun transcribeAudio(audioFile: File): String {
        val audioRequestBody = audioFile.asRequestBody("audio/*".toMediaType())

        val formBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("file", audioFile.name, audioRequestBody)
            .addFormDataPart("model", "whisper-1")
            .addFormDataPart("language", "en")
            .build()

        val request = Request.Builder()
            .url("https://api.openai.com/v1/audio/transcriptions")
            .header("Authorization", "Bearer $API_KEY")
            .post(formBody)

        return client.newCall(request.build()).execute().use { response ->
            response.body?.string() ?: ""
        }
    }

    private fun createChatCompletion(prompt: String): String {

        val mediaType = "application/json; charset=utf-8".toMediaType()
        val completeString = "I say $prompt. As an assistant how can you help me?\n" +
                "Pick one from the options below if it is related to volume and write only the two words:\n" +
                "volume up\n" +
                "volume down\n" +
                "if it is not related to volume answer the below two words: \n" +
                "volume stable"

        val messagesArray = JSONArray()
        messagesArray.put(JSONObject().put("role", "system").put("content", "You are a helpful assistant inside a car."))
        // messagesArray.put(JSONObject().put("role", "user").put("content", "Who won the world series in 2020?"))
        // messagesArray.put(JSONObject().put("role", "assistant").put("content", "The Los Angeles Dodgers won the World Series in 2020."))
        messagesArray.put(JSONObject().put("role", "user").put("content", completeString))

        val json = JSONObject()
            .put("model", "gpt-3.5-turbo")
            .put("messages", messagesArray)

        val requestBody = json.toString().toRequestBody(mediaType)
        val request = Request.Builder()
            .url("https://api.openai.com/v1/chat/completions")
            .header("Authorization", "Bearer $API_KEY")
            .post(requestBody)
            .build()

        val gson = Gson()

        client.newCall(request).execute().use { response ->
            val chatCompletionResponse = gson.fromJson(response.body?.string() ?: "", GeneratedAnswer::class.java)
            return chatCompletionResponse.choices?.get(0)?.message?.content.toString()
        }
    }

    override fun onCleared() {
        super.onCleared()
        mediaRecorder.release()
        audioClassificationHelper.stopAudioClassification()
    }

    companion object {

    }
}

interface AudioClassificationListener {
    fun onError(error: String)
    fun onResult(results: List<Category>, inferenceTime: Long)
}
