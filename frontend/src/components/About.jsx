import { Box, Container, Typography, Paper, Grid } from '@mui/material'
import CodeIcon from '@mui/icons-material/Code'
import SecurityIcon from '@mui/icons-material/Security'
import SpeedIcon from '@mui/icons-material/Speed'

const About = () => {
  const features = [
    {
      icon: <CodeIcon sx={{ fontSize: 40, color: 'primary.main' }} />,
      title: 'Advanced Technology',
      description:
        'Built with cutting-edge OCR technology and modern web frameworks including FastAPI and React.',
    },
    {
      icon: <SecurityIcon sx={{ fontSize: 40, color: 'primary.main' }} />,
      title: 'Secure Processing',
      description:
        'Your documents are processed securely and are never stored permanently on our servers.',
    },
    {
      icon: <SpeedIcon sx={{ fontSize: 40, color: 'primary.main' }} />,
      title: 'Fast & Efficient',
      description:
        'Optimized for speed and efficiency, providing quick results for all your document processing needs.',
    },
  ]

  return (
    <Box sx={{ py: 8, backgroundColor: 'background.default' }}>
      <Container maxWidth="lg">
        <Box textAlign="center" mb={8}>
          <Typography
            variant="h3"
            component="h1"
            gutterBottom
            sx={{
              fontWeight: 'bold',
              color: 'primary.main',
            }}
          >
            About OCR Pro
          </Typography>
          <Typography variant="h6" color="text.secondary" sx={{ mb: 4 }}>
            Your All-in-One Solution for Text Extraction and Processing
          </Typography>
        </Box>

        <Grid container spacing={4} sx={{ mb: 8 }}>
          {features.map((feature, index) => (
            <Grid item xs={12} md={4} key={index}>
              <Paper
                elevation={3}
                sx={{
                  p: 4,
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  textAlign: 'center',
                }}
              >
                {feature.icon}
                <Typography variant="h6" sx={{ mt: 2, mb: 1 }}>
                  {feature.title}
                </Typography>
                <Typography color="text.secondary">
                  {feature.description}
                </Typography>
              </Paper>
            </Grid>
          ))}
        </Grid>

        <Paper elevation={1} sx={{ p: 4, backgroundColor: 'rgba(25, 118, 210, 0.04)' }}>
          <Typography variant="h5" gutterBottom>
            Our Mission
          </Typography>
          <Typography paragraph>
            OCR Pro aims to simplify document processing by providing powerful OCR capabilities
            combined with advanced language processing features. We strive to make text
            extraction, translation, and summarization accessible to everyone.
          </Typography>
          <Typography variant="h5" gutterBottom sx={{ mt: 4 }}>
            Technology Stack
          </Typography>
          <Typography paragraph>
            Our application is built using modern technologies including:
          </Typography>
          <ul>
            <Typography component="li">
              Frontend: React with Material-UI for a responsive and intuitive interface
            </Typography>
            <Typography component="li">
              Backend: FastAPI for high-performance API endpoints
            </Typography>
            <Typography component="li">
              OCR: Advanced OCR technology for accurate text extraction
            </Typography>
            <Typography component="li">
              AI: State-of-the-art models for text summarization and translation
            </Typography>
          </ul>
        </Paper>
      </Container>
    </Box>
  )
}

export default About 