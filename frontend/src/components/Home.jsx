import React from 'react'
import { Box, Container, Typography, Paper, Button, Card, CardContent, CardActions, Grid } from '@mui/material'
import TextSnippetIcon from '@mui/icons-material/TextSnippet'
import TranslateIcon from '@mui/icons-material/Translate'
import SummarizeIcon from '@mui/icons-material/Summarize'
import ImageIcon from '@mui/icons-material/Image'
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf'
import AutoStoriesIcon from '@mui/icons-material/AutoStories'
import { useNavigate } from 'react-router-dom'
import { keyframes } from '@mui/system'

// Define animations
const fadeInUp = keyframes`
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
`

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

const FeatureCard = ({ icon, title, description, path, index }) => {
  const navigate = useNavigate()
  return (
    <Card 
      sx={{ 
        width: '100%',
        height: '100%',
        minHeight: 400,
        display: 'flex', 
        flexDirection: 'column',
        transition: 'all 0.3s ease-in-out',
        animation: `${fadeInUp} 0.6s ease-out forwards ${index * 0.1}s`,
        opacity: 0,
        background: 'linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%)',
        borderRadius: 3,
        overflow: 'hidden',
        '&:hover': {
          transform: 'translateY(-8px)',
          boxShadow: '0 12px 20px rgba(0,0,0,0.1)',
          '& .icon': {
            animation: `${pulse} 0.5s ease-in-out`,
          },
        },
      }}
    >
      <CardContent 
        sx={{ 
          flexGrow: 1,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'space-between',
          p: 4,
          gap: 3,
        }}
      >
        <Box 
          sx={{ 
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: 3,
            width: '100%',
          }}
        >
          <Box 
            className="icon"
            sx={{
              display: 'flex',
              justifyContent: 'center',
              width: '100%',
              mb: 1,
            }}
          >
            {React.cloneElement(icon, { 
              sx: { 
                fontSize: 64,
                color: 'primary.main',
                filter: 'drop-shadow(0 4px 6px rgba(0,0,0,0.1))',
              }, 
            })}
          </Box>
          <Typography 
            variant="h5" 
            component="h3" 
            sx={{ 
              fontWeight: 600,
              background: 'linear-gradient(45deg, #1976d2, #21CBF3)',
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              color: 'transparent',
              textAlign: 'center',
              width: '100%',
            }}
          >
            {title}
          </Typography>
          <Typography 
            variant="body1" 
            color="text.secondary"
            sx={{ 
              textAlign: 'center',
              height: '4.8em',
              display: '-webkit-box',
              WebkitLineClamp: 3,
              WebkitBoxOrient: 'vertical',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              lineHeight: 1.6,
              width: '100%',
              maxWidth: '90%',
              margin: '0 auto',
            }}
          >
            {description}
          </Typography>
        </Box>
      </CardContent>
      <CardActions 
        sx={{ 
          justifyContent: 'center',
          p: 4,
          pt: 0,
          width: '100%',
        }}
      >
        <Button 
          variant="contained" 
          onClick={() => navigate(path)}
          sx={{
            px: 4,
            py: 1.5,
            borderRadius: 3,
            textTransform: 'none',
            fontSize: '1.1rem',
            background: 'linear-gradient(45deg, #1976d2, #21CBF3)',
            boxShadow: '0 4px 10px rgba(0,0,0,0.15)',
            minWidth: '160px',
            '&:hover': {
              background: 'linear-gradient(45deg, #1565c0, #1976d2)',
              transform: 'translateY(-2px)',
              boxShadow: '0 6px 12px rgba(0,0,0,0.2)',
            },
          }}
        >
          Try Now
        </Button>
      </CardActions>
    </Card>
  )
}

const Home = () => {
  const features = [
    {
      icon: <TextSnippetIcon />,
      title: 'Text Extraction',
      description: 'Extract text from any document or image with high accuracy using advanced OCR technology.',
      path: '/extract',
    },
    {
      icon: <SummarizeIcon />,
      title: 'Text Summarization',
      description: 'Get concise summaries of long documents using AI-powered summarization.',
      path: '/summarize',
    },
    {
      icon: <TranslateIcon />,
      title: 'Translation',
      description: 'Translate extracted text into multiple languages with reliable accuracy.',
      path: '/translate',
    },
    {
      icon: <ImageIcon />,
      title: 'Image to Text',
      description: 'Convert images containing text into editable text format with OCR.',
      path: '/image-to-text',
    },
    {
      icon: <PictureAsPdfIcon />,
      title: 'PDF Tools',
      description: 'Extract text from PDFs, convert PDFs to text, and more.',
      path: '/pdf-tools',
    },
    {
      icon: <AutoStoriesIcon />,
      title: 'Document Analysis',
      description: 'Analyze documents for key information and extract structured data.',
      path: '/document-analysis',
    },
  ]

  return (
    <Box sx={{ backgroundColor: 'background.default' }}>
      {/* Hero Section */}
      <Box 
        sx={{ 
          background: 'linear-gradient(135deg, #1976d2 0%, #21CBF3 100%)',
          color: 'white',
          py: { xs: 8, md: 12 },
          mb: 6,
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
          },
        }}
      >
        <Container maxWidth="lg">
          <Grid container spacing={4} alignItems="center">
            <Grid item xs={12} md={8} sx={{ 
              animation: `${fadeInUp} 0.8s ease-out`,
            }}>
              <Typography 
                variant="h1" 
                component="h1" 
                gutterBottom 
                sx={{ 
                  fontWeight: 800,
                  fontSize: { xs: '2.5rem', md: '3.5rem' },
                  textShadow: '0 2px 4px rgba(0,0,0,0.1)',
                  mb: 3,
                }}
              >
                Smart Document Processing
              </Typography>
              <Typography 
                variant="h5" 
                paragraph
                sx={{ 
                  opacity: 0.9,
                  mb: 4,
                  maxWidth: '600px',
                  lineHeight: 1.6,
                }}
              >
                Transform your documents into actionable insights with our AI-powered tools
              </Typography>
            </Grid>
          </Grid>
        </Container>
      </Box>

      {/* Project Information */}
      <Container maxWidth="lg">
        <Box sx={{ mb: 8 }}>
          <Typography 
            variant="h3" 
            component="h2" 
            gutterBottom 
            sx={{ 
              mb: 4,
              fontWeight: 700,
              textAlign: 'center',
              background: 'linear-gradient(45deg, #1976d2, #21CBF3)',
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              color: 'transparent',
            }}
          >
            About Our Project
          </Typography>
          
          <Grid container spacing={4}>
            {[
              {
                text: 'OCR Pro is a comprehensive document processing solution that combines cutting-edge OCR technology with advanced language processing capabilities. Our platform is designed to handle various document formats and provide accurate text extraction, translation, and summarization services.',
              },
              {
                text: 'We leverage state-of-the-art artificial intelligence and machine learning models to ensure high accuracy in text recognition and processing. Our system supports multiple languages and can handle complex document layouts while maintaining the original formatting structure.',
              },
              {
                text: 'Whether you\'re a business professional looking to digitize documents, a researcher needing to extract text from papers, or anyone working with document processing, OCR Pro provides the tools you need to streamline your workflow and increase productivity.',
              },
            ].map((item, index) => (
              <Grid item xs={12} md={4} key={index}>
                <Paper 
                  elevation={0} 
                  sx={{ 
                    p: 4,
                    height: '100%',
                    backgroundColor: 'rgba(25, 118, 210, 0.04)',
                    borderRadius: 3,
                    transition: 'all 0.3s ease',
                    '&:hover': {
                      transform: 'translateY(-4px)',
                      boxShadow: '0 6px 12px rgba(0,0,0,0.1)',
                      backgroundColor: 'rgba(25, 118, 210, 0.06)',
                    },
                  }}
                >
                  <Typography 
                    paragraph
                    sx={{
                      fontSize: '1.1rem',
                      lineHeight: 1.7,
                      color: 'text.secondary',
                    }}
                  >
                    {item.text}
                  </Typography>
                </Paper>
              </Grid>
            ))}
          </Grid>
        </Box>

        {/* Features Grid */}
        <Container maxWidth="lg" sx={{ mb: 8 }}>
          <Grid 
            container 
            spacing={4} 
            sx={{ 
              display: 'grid',
              gridTemplateColumns: {
                xs: '1fr',
                sm: 'repeat(2, 1fr)',
                md: 'repeat(3, 1fr)',
              },
              gap: 4,
              alignItems: 'stretch',
            }}
          >
            {features.map((feature, index) => (
              <Grid item key={feature.title} sx={{ height: '100%', display: 'flex' }}>
                <FeatureCard {...feature} index={index} />
              </Grid>
            ))}
          </Grid>
        </Container>
      </Container>
    </Box>
  )
}

export default Home 