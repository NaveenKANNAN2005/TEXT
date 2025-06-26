import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { Container, CssBaseline, ThemeProvider, createTheme, Typography, Box, Paper } from '@mui/material'
import { ToastContainer } from 'react-toastify'
import 'react-toastify/dist/ReactToastify.css'
import Navbar from './components/Navbar'
import Home from './components/Home'
import FileUpload from './components/FileUpload'
import ResultDisplay from './components/ResultDisplay'
import { useState } from 'react'
import { keyframes } from '@mui/system'

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

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 700,
      '@media (min-width:600px)': {
        fontSize: '3rem',
      },
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 8,
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 12,
        },
      },
    },
  },
})

// Feature page component template
const FeaturePage = ({ title, description }) => {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  return (
    <Box sx={{ py: 4, animation: `${fadeIn} 0.6s ease-out` }}>
      <Container maxWidth="lg">
        <Paper 
          elevation={0} 
          sx={{ 
            p: { xs: 3, md: 6 },
            mb: 4,
            background: 'linear-gradient(135deg, #1976d2 0%, #21CBF3 100%)',
            color: 'white',
            position: 'relative',
            overflow: 'hidden',
            '&::after': {
              content: '""',
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              background: 'radial-gradient(circle at 20% 150%, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 50%)',
            }
          }}
        >
          <Typography 
            variant="h1" 
            component="h1" 
            gutterBottom
            sx={{
              fontWeight: 800,
              textShadow: '0 2px 4px rgba(0,0,0,0.1)',
              position: 'relative',
              zIndex: 1,
            }}
          >
            {title}
          </Typography>
          <Typography 
            variant="h5" 
            sx={{ 
              mb: 2,
              opacity: 0.9,
              maxWidth: '800px',
              lineHeight: 1.6,
              position: 'relative',
              zIndex: 1,
            }}
          >
            {description}
          </Typography>
        </Paper>

        <Paper 
          elevation={2} 
          sx={{ 
            p: { xs: 2, md: 4 },
            backgroundColor: 'white',
            transition: 'transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out',
            '&:hover': {
              transform: 'translateY(-4px)',
              boxShadow: '0 8px 16px rgba(0,0,0,0.1)',
            },
          }}
        >
          <FileUpload setResult={setResult} setLoading={setLoading} />
          <ResultDisplay result={result} loading={loading} />
        </Paper>
      </Container>
    </Box>
  )
}

function App() {
  return (
    <Router>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
          <Navbar />
          <main style={{ flex: 1 }}>
            <Routes>
              <Route path="/" element={<Home />} />
              <Route
                path="/extract"
                element={
                  <FeaturePage
                    title="Text Extraction"
                    description="Extract text from any document with high accuracy using our advanced OCR technology. Support for multiple file formats and languages."
                  />
                }
              />
              <Route
                path="/summarize"
                element={
                  <FeaturePage
                    title="Text Summarization"
                    description="Get concise and accurate summaries of your documents using our AI-powered technology. Perfect for long documents and research papers."
                  />
                }
              />
              <Route
                path="/translate"
                element={
                  <FeaturePage
                    title="Translation"
                    description="Translate your documents into multiple languages with high accuracy. Supporting a wide range of languages and maintaining formatting."
                  />
                }
              />
              <Route
                path="/image-to-text"
                element={
                  <FeaturePage
                    title="Image to Text"
                    description="Convert images containing text into editable format using advanced OCR. Support for multiple image formats and complex layouts."
                  />
                }
              />
              <Route
                path="/pdf-tools"
                element={
                  <FeaturePage
                    title="PDF Tools"
                    description="Comprehensive suite of PDF processing tools. Extract text, convert formats, and maintain document structure with ease."
                  />
                }
              />
              <Route
                path="/document-analysis"
                element={
                  <FeaturePage
                    title="Document Analysis"
                    description="Advanced document analysis and information extraction. Identify key information and extract structured data automatically."
                  />
                }
              />
            </Routes>
          </main>
        </div>
        <ToastContainer 
          position="bottom-right"
          autoClose={5000}
          hideProgressBar={false}
          newestOnTop
          closeOnClick
          rtl={false}
          pauseOnFocusLoss
          draggable
          pauseOnHover
          theme="colored"
        />
      </ThemeProvider>
    </Router>
  )
}

export default App
