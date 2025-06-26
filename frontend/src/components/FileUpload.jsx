import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Box, Button, Typography, CircularProgress, Paper, Select, MenuItem, FormControl, InputLabel } from '@mui/material'
import CloudUploadIcon from '@mui/icons-material/CloudUpload'
import { toast } from 'react-toastify'
import { keyframes } from '@mui/system'
import { useLocation } from 'react-router-dom'

const pulse = keyframes`
  0% {
    transform: scale(1);
  }
  50% {
    transform: scale(1.05);
  }
  100% {
    transform: scale(1);
  }
`

const FileUpload = ({ setResult, setLoading }) => {
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [targetLanguage, setTargetLanguage] = useState('en')
  const location = useLocation()

  // Get current feature from pathname
  const currentFeature = location.pathname.substring(1)

  // Define accepted file types based on feature
  const getAcceptedFiles = () => {
    switch (currentFeature) {
      case 'image-to-text':
        return {
          'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
        }
      case 'pdf-tools':
        return {
          'application/pdf': ['.pdf']
        }
      default:
        return {
          'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.bmp'],
          'application/pdf': ['.pdf'],
          'text/plain': ['.txt'],
          'application/msword': ['.doc'],
          'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx']
        }
    }
  }

  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      setFile(acceptedFiles[0])
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: getAcceptedFiles(),
    maxSize: 10485760, // 10MB
    multiple: false
  })

  const handleUpload = async () => {
    if (!file) {
      toast.error('Please select a file first')
      return
    }

    const formData = new FormData()
    formData.append('file', file)

    // Add feature-specific parameters
    switch (currentFeature) {
      case 'summarize':
        formData.append('summarize', 'true')
        formData.append('languages', 'en')
        formData.append('target_language', 'en')
        formData.append('mode', 'summarize')
        break
      case 'translate':
        formData.append('summarize', 'false')
        formData.append('languages', 'en')
        formData.append('target_language', targetLanguage)
        formData.append('mode', 'translate')
        break
      case 'image-to-text':
        formData.append('summarize', 'false')
        formData.append('languages', 'en')
        formData.append('target_language', 'en')
        formData.append('mode', 'image-to-text')
        break
      case 'pdf-tools':
        formData.append('summarize', 'false')
        formData.append('languages', 'en')
        formData.append('target_language', 'en')
        formData.append('mode', 'pdf')
        break
      case 'document-analysis':
        formData.append('summarize', 'false')
        formData.append('languages', 'en')
        formData.append('target_language', 'en')
        formData.append('mode', 'analysis')
        break
      default:
        formData.append('summarize', 'false')
        formData.append('languages', 'en')
        formData.append('target_language', 'en')
        formData.append('mode', 'extract')
    }

    setUploading(true)
    setLoading(true)

    try {
      const response = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Upload failed')
      }

      const data = await response.json()
      setResult(data)
      toast.success('File processed successfully!')
    } catch (error) {
      console.error('Error:', error)
      toast.error('Failed to process file. Please try again.')
      setResult(null)
    } finally {
      setUploading(false)
      setLoading(false)
    }
  }

  // Get helper text based on current feature
  const getHelperText = () => {
    switch (currentFeature) {
      case 'image-to-text':
        return 'Supported formats: Images (PNG, JPG, GIF)'
      case 'pdf-tools':
        return 'Supported formats: PDF'
      case 'translate':
        return 'Supported formats: PDF, Images, Word (DOC, DOCX), TXT'
      default:
        return 'Supported formats: PDF, Images (PNG, JPG, GIF), Word (DOC, DOCX), TXT'
    }
  }

  return (
    <Box sx={{ textAlign: 'center', mb: 4 }}>
      <Paper
        {...getRootProps()}
        elevation={0}
        sx={{
          p: 6,
          mb: 3,
          cursor: 'pointer',
          backgroundColor: isDragActive ? 'rgba(25, 118, 210, 0.08)' : 'rgba(25, 118, 210, 0.04)',
          border: '2px dashed',
          borderColor: isDragActive ? 'primary.main' : 'rgba(0, 0, 0, 0.12)',
          borderRadius: 3,
          transition: 'all 0.3s ease-in-out',
          position: 'relative',
          overflow: 'hidden',
          '&:hover': {
            backgroundColor: 'rgba(25, 118, 210, 0.08)',
            borderColor: 'primary.main',
            '& .upload-icon': {
              animation: `${pulse} 1s ease-in-out infinite`,
            },
          },
        }}
      >
        <input {...getInputProps()} />
        <Box className="upload-icon" sx={{ mb: 2 }}>
          <CloudUploadIcon
            sx={{
              fontSize: 64,
              color: isDragActive ? 'primary.main' : 'action.active',
              transition: 'color 0.3s ease',
            }}
          />
        </Box>
        <Typography
          variant="h5"
          component="div"
          gutterBottom
          sx={{
            fontWeight: 600,
            color: isDragActive ? 'primary.main' : 'text.primary',
            transition: 'color 0.3s ease',
          }}
        >
          {isDragActive ? 'Drop the file here' : 'Drag & drop your file here'}
        </Typography>
        <Typography
          variant="body1"
          color="text.secondary"
          sx={{ mb: 2 }}
        >
          {file ? `Selected file: ${file.name}` : 'or click to select a file'}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          {getHelperText()}
          <br />
          Maximum file size: 10MB
        </Typography>
      </Paper>

      {currentFeature === 'translate' && (
        <FormControl sx={{ mb: 3, minWidth: 200 }}>
          <InputLabel id="target-language-label">Target Language</InputLabel>
          <Select
            labelId="target-language-label"
            value={targetLanguage}
            label="Target Language"
            onChange={(e) => setTargetLanguage(e.target.value)}
          >
            <MenuItem value="en">English</MenuItem>
            <MenuItem value="es">Spanish</MenuItem>
            <MenuItem value="fr">French</MenuItem>
            <MenuItem value="de">German</MenuItem>
            <MenuItem value="it">Italian</MenuItem>
            <MenuItem value="pt">Portuguese</MenuItem>
            <MenuItem value="ru">Russian</MenuItem>
            <MenuItem value="zh">Chinese</MenuItem>
            <MenuItem value="ja">Japanese</MenuItem>
            <MenuItem value="ko">Korean</MenuItem>
            <MenuItem value="ta">Tamil</MenuItem>
            <MenuItem value="hi">Hindi</MenuItem>
          </Select>
        </FormControl>
      )}

      <Button
        variant="contained"
        onClick={handleUpload}
        disabled={!file || uploading}
        sx={{
          px: 4,
          py: 1.5,
          fontSize: '1.1rem',
          background: 'linear-gradient(45deg, #1976d2, #21CBF3)',
          boxShadow: '0 4px 10px rgba(0,0,0,0.15)',
          '&:hover': {
            background: 'linear-gradient(45deg, #1565c0, #1976d2)',
            transform: 'translateY(-2px)',
            boxShadow: '0 6px 12px rgba(0,0,0,0.2)',
          },
          '&:disabled': {
            background: '#ccc',
          },
        }}
      >
        {uploading ? (
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <CircularProgress size={24} sx={{ mr: 1, color: 'white' }} />
            Processing...
          </Box>
        ) : (
          'Process File'
        )}
      </Button>
    </Box>
  )
}

export default FileUpload 