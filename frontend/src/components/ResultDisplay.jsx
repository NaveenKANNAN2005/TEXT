import { Box, Typography, Paper, CircularProgress, Button, Divider } from '@mui/material'
import ContentCopyIcon from '@mui/icons-material/ContentCopy'
import CheckCircleIcon from '@mui/icons-material/CheckCircle'
import { useState } from 'react'
import { toast } from 'react-toastify'
import { keyframes } from '@mui/system'
import { useLocation } from 'react-router-dom'

const fadeIn = keyframes`
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
`

const shimmer = keyframes`
  0% {
    background-position: -1000px 0;
  }
  100% {
    background-position: 1000px 0;
  }
`

const ResultDisplay = ({ result, loading }) => {
  const [copied, setCopied] = useState(false)
  const location = useLocation()
  const currentFeature = location.pathname.substring(1)

  const handleCopy = async (text) => {
    if (text) {
      try {
        await navigator.clipboard.writeText(text)
        setCopied(true)
        toast.success('Text copied to clipboard!')
        setTimeout(() => setCopied(false), 2000)
      } catch (err) {
        toast.error('Failed to copy text')
      }
    }
  }

  if (loading) {
    return (
      <Paper
        elevation={0}
        sx={{
          p: 4,
          textAlign: 'center',
          backgroundColor: 'rgba(25, 118, 210, 0.04)',
          borderRadius: 3,
          animation: `${fadeIn} 0.5s ease-out`,
        }}
      >
        <CircularProgress
          size={60}
          thickness={4}
          sx={{
            mb: 3,
            color: 'primary.main',
          }}
        />
        <Typography
          variant="h6"
          sx={{
            fontWeight: 600,
            background: 'linear-gradient(45deg, #1976d2, #21CBF3)',
            backgroundClip: 'text',
            WebkitBackgroundClip: 'text',
            color: 'transparent',
          }}
        >
          {currentFeature === 'translate' ? 'Translating your text...' :
           currentFeature === 'summarize' ? 'Generating summary...' :
           currentFeature === 'image-to-text' ? 'Extracting text from image...' :
           currentFeature === 'pdf-tools' ? 'Processing PDF...' :
           currentFeature === 'document-analysis' ? 'Analyzing document...' :
           'Processing your file...'}
        </Typography>
        <Typography
          variant="body1"
          color="text.secondary"
          sx={{ mt: 1 }}
        >
          This may take a few moments
        </Typography>
      </Paper>
    )
  }

  if (!result) return null

  const renderContent = () => {
    switch (currentFeature) {
      case 'translate':
        return (
          <>
            <Box sx={{ mb: 3 }}>
              <Typography variant="h6" gutterBottom>Original Text</Typography>
              <Paper elevation={0} sx={{ p: 3, backgroundColor: 'rgba(25, 118, 210, 0.04)', borderRadius: 2 }}>
                <Typography variant="body1">{result.text}</Typography>
              </Paper>
            </Box>
            <Divider sx={{ my: 3 }} />
            <Box>
              <Typography variant="h6" gutterBottom>Translated Text</Typography>
              <Paper elevation={0} sx={{ p: 3, backgroundColor: 'rgba(25, 118, 210, 0.04)', borderRadius: 2 }}>
                <Typography variant="body1">{result.translated_text}</Typography>
              </Paper>
            </Box>
          </>
        )

      case 'summarize':
        return (
          <Box>
            <Typography variant="h6" gutterBottom>Summary</Typography>
            <Paper elevation={0} sx={{ p: 3, backgroundColor: 'rgba(25, 118, 210, 0.04)', borderRadius: 2 }}>
              <Typography variant="body1">{result.summary}</Typography>
            </Paper>
          </Box>
        )

      case 'document-analysis':
        const renderValue = (value) => {
          if (Array.isArray(value)) {
            return (
              <Box component="ul" sx={{ mt: 0.5, mb: 1, pl: 2 }}>
                {value.map((item, index) => (
                  <Typography component="li" key={index} variant="body2">
                    {typeof item === 'object' && item.word ? 
                      `${item.word}: ${item.frequency} occurrences` : 
                      String(item)}
                  </Typography>
                ))}
              </Box>
            );
          }
          
          if (typeof value === 'object' && value !== null) {
            return Object.entries(value).map(([subKey, subValue]) => (
              <Box key={subKey} sx={{ ml: 2, mb: 1 }}>
                <Typography variant="subtitle2" sx={{ fontWeight: 500 }}>
                  {subKey.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}:
                </Typography>
                <Box sx={{ ml: 2 }}>
                  {renderValue(subValue)}
                </Box>
              </Box>
            ));
          }
          
          return (
            <Typography variant="body2">
              {typeof value === 'number' ? 
                (Number.isInteger(value) ? value : value.toFixed(2)) : 
                String(value)}
            </Typography>
          );
        };

        return (
          <>
            <Box sx={{ mb: 3 }}>
              <Typography variant="h6" gutterBottom>Extracted Text</Typography>
              <Paper elevation={0} sx={{ p: 3, backgroundColor: 'rgba(25, 118, 210, 0.04)', borderRadius: 2 }}>
                <Typography variant="body1">{result.text}</Typography>
              </Paper>
            </Box>
            <Divider sx={{ my: 3 }} />
            <Box>
              <Typography variant="h6" gutterBottom>Analysis</Typography>
              <Paper elevation={0} sx={{ p: 3, backgroundColor: 'rgba(25, 118, 210, 0.04)', borderRadius: 2 }}>
                {result.analysis && Object.entries(result.analysis).map(([key, value]) => (
                  <Box key={key} sx={{ mb: 3 }}>
                    <Typography variant="subtitle1" sx={{ fontWeight: 600, color: 'primary.main', mb: 1 }}>
                      {key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
                    </Typography>
                    <Box sx={{ ml: 2 }}>
                      {renderValue(value)}
                    </Box>
                  </Box>
                ))}
              </Paper>
            </Box>
          </>
        )

      default:
        return (
          <Paper
            elevation={0}
            sx={{
              p: 3,
              backgroundColor: 'rgba(25, 118, 210, 0.04)',
              borderRadius: 2,
              maxHeight: '400px',
              overflowY: 'auto',
            }}
          >
            <Typography
              variant="body1"
              component="pre"
              sx={{
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
                fontFamily: 'inherit',
                margin: 0,
                lineHeight: 1.6,
              }}
            >
              {result.text}
              {result.confidence && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    Confidence: {(result.confidence * 100).toFixed(2)}%
                  </Typography>
                </Box>
              )}
            </Typography>
          </Paper>
        )
    }
  }

  return (
    <Paper
      elevation={0}
      sx={{
        p: 4,
        backgroundColor: 'white',
        borderRadius: 3,
        animation: `${fadeIn} 0.5s ease-out`,
        border: '1px solid',
        borderColor: 'divider',
      }}
    >
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography
          variant="h5"
          component="h3"
          sx={{
            fontWeight: 600,
            background: 'linear-gradient(45deg, #1976d2, #21CBF3)',
            backgroundClip: 'text',
            WebkitBackgroundClip: 'text',
            color: 'transparent',
          }}
        >
          Results
        </Typography>
        <Button
          variant="outlined"
          startIcon={copied ? <CheckCircleIcon /> : <ContentCopyIcon />}
          onClick={() => handleCopy(currentFeature === 'translate' ? result.translated_text : result.text)}
          disabled={!result?.text}
          sx={{
            borderRadius: 2,
            textTransform: 'none',
            transition: 'all 0.3s ease',
            borderColor: copied ? 'success.main' : 'primary.main',
            color: copied ? 'success.main' : 'primary.main',
            '&:hover': {
              borderColor: copied ? 'success.dark' : 'primary.dark',
              backgroundColor: copied ? 'success.light' : 'primary.light',
              opacity: 0.1,
            },
          }}
        >
          {copied ? 'Copied!' : 'Copy Text'}
        </Button>
      </Box>

      {renderContent()}
    </Paper>
  )
}

export default ResultDisplay 