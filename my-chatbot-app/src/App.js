import React, { useState } from 'react';
import { Container, Grid } from '@mui/material';
import ChatWindow from './components/ChatWindow';
import axios from 'axios'; // Import axios for making HTTP requests
import './App.css'; // Import CSS file for styling

const App = () => {
  const [messages, setMessages] = useState([]);

  const handleSendMessage = async (message) => {
    const userMessage = { text: message, isUserMessage: true }; // Create user message object
    setMessages((prevMessages) => [...prevMessages, userMessage]); // Add user message to UI

    try {
      const response = await axios.post('http://localhost:8000/api/chatbot', { input_text: message }); // Send message to FastAPI server
      const responseData = response.data;
      const systemResponse = { text: responseData.response, isUserMessage: false, withAvatar: true }; // Create system response object
      setMessages((prevMessages) => [...prevMessages, systemResponse]); // Add system response to UI
    } catch (error) {
      console.error('Error sending message:', error);
    }
  };

  return (
    <div className="wrapper">
  <div className="circle xxlarge shade1"></div>
  <div className="circle xlarge shade2"></div>
  <div className="circle large shade3"></div>
  <div className="circle medium shade4"></div>
  <div className="circle small shade5"></div>

  <Container maxWidth="lg">
    <Grid container spacing={2}>
      <Grid item xs={12} md={8} className="chat-container">
        <ChatWindow messages={messages} onSendMessage={handleSendMessage} />
      </Grid>
      
    </Grid>
  </Container>
</div>  );
};

export default App;
