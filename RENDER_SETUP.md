# Render Deployment Setup for EthoScore Backend

## Environment Variables Required

You need to set up the following environment variables in your Render dashboard:

### Your Environment Variables (From your .env file)

Based on your existing setup, add these to Render:

```
MODEL_URL_ORDINAL_MODEL_BEST_CHECKPOINT_SAFETENSORS=https://drive.google.com/uc?export=download&id=1Ed0s_QInYFOgh84YvATcBIFy-ksN9hKr

MODEL_URL_3CLASS_MODEL_BEST_CHECKPOINT_SAFETENSORS=https://drive.google.com/uc?export=download&id=1D6JJyFbrJNPq4s__wGQAh37EL0WUAvEq

MODEL_URL_DATASET_FRAMING_ANNOTATIONS_LLAMA_3_3_70B_INSTRUCT_TURBO_CSV=https://drive.google.com/uc?export=download&id=18WPDWCNONixCAVJs_GxkJHm10mbCr9H4
```

### Alternative Formats (Also Supported)

The code also supports these shorter variable names:

**Option 1: File IDs only**
```
ORDINAL_MODEL_ID=1Ed0s_QInYFOgh84YvATcBIFy-ksN9hKr
CLASS_3_MODEL_ID=1D6JJyFbrJNPq4s__wGQAh37EL0WUAvEq
DATASET_ID=18WPDWCNONixCAVJs_GxkJHm10mbCr9H4
```

**Option 2: Full URLs**
```
ORDINAL_MODEL_URL=https://drive.google.com/uc?export=download&id=1Ed0s_QInYFOgh84YvATcBIFy-ksN9hKr
CLASS_3_MODEL_URL=https://drive.google.com/uc?export=download&id=1D6JJyFbrJNPq4s__wGQAh37EL0WUAvEq
DATASET_URL=https://drive.google.com/uc?export=download&id=18WPDWCNONixCAVJs_GxkJHm10mbCr9H4
```

## How to Get Google Drive File IDs

1. Upload your model files to Google Drive
2. Right-click on each file → "Get link"
3. Make sure the link is set to "Anyone with the link can view"
4. The URL will look like: `https://drive.google.com/file/d/FILE_ID_HERE/view`
5. Copy the `FILE_ID_HERE` part

## Files That Need to Be Uploaded

Make sure you have these three files in Google Drive:

1. **ordinal_model_best_checkpoint.safetensors** - The ordinal classification model
2. **3class_model_best_checkpoint.safetensors** - The 3-class classification model  
3. **Dataset-framing_annotations-Llama-3.3-70B-Instruct-Turbo.csv** - The training dataset

## Important: Make Files Public

For the download to work without authentication, you MUST:

1. Right-click each file in Google Drive
2. Click "Share"
3. Change "Restricted" to "Anyone with the link"
4. Click "Done"

### Large File Virus Scan Warning

Google Drive shows a warning for large files (>100MB):
> "Google Drive can't scan this file for viruses. Would you still like to download this file?"

**This is normal and expected!** The backend code automatically handles this warning and bypasses it. Your files are safe - they're model files you uploaded yourself.

## Deployment Steps

1. Set up all environment variables in Render
2. Trigger a new deployment
3. The server will automatically download the model files on startup
4. Check the logs to verify successful download
5. Visit `/health` endpoint to verify models are initialized

## Verifying Deployment

After deployment, check:

```bash
curl https://your-app.onrender.com/health
```

Should return:
```json
{
  "ok": true,
  "models": {
    "is_initialized": true
  },
  "dataset_loaded": true
}
```

## Troubleshooting

If models don't load:

1. **Check environment variables** - Make sure they're set correctly in Render
2. **Check file permissions** - Ensure Google Drive files are public
3. **Check logs** - Look for download errors in Render logs
4. **File size** - Large files may take time to download on first startup
5. **Timeout** - Render may timeout if download takes too long. Consider using smaller model files or a different hosting solution for very large files.

## Alternative: Direct Upload (Not Recommended)

If Google Drive download doesn't work, you can include the model files in your git repository:

1. Use Git LFS to track the large files:
   ```bash
   git lfs track "*.safetensors"
   git lfs track "*.csv"
   git add .gitattributes
   ```
2. Commit and push the model files
3. Render will download them during build

⚠️ **Note**: This will make your repository very large and slow to clone.
