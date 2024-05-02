import React, { useState, useEffect, useRef } from 'react';
import { Typography, TextField, Button, Paper, Chip } from '@mui/material';
import Avatar from '@mui/material/Avatar';
import Fade from '@mui/material/Fade';
import Badge from '@mui/material/Badge';

const ChatMessage = ({ message }) => {
  return (
    <Fade in={true} timeout={500}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: message.isUserMessage ? 'flex-end' : 'flex-start',
          marginBottom: 10,
        }}
      >
        <div
          style={{
            backgroundColor: message.isUserMessage ? '#E3F2FD' : '#BBDEFB',
            padding: 10,
            borderRadius: 20,
            maxWidth: '70%',
          }}
        >
          <Typography variant="body1">{message.text}</Typography>
        </div>
        {message.isUserMessage && (
          <Badge
            overlap="circular"
            anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
            badgeContent={<div style={{ backgroundColor: 'green', width: 10, height: 10, borderRadius: '50%' }} />}
          >
            <Avatar alt="User Avatar" src="../../public/avatar.png" style={{ marginLeft: 10 }} />
          </Badge>
        )}
      </div>
    </Fade>
  );
};

const ChatWindow = ({ messages, onSendMessage }) => {
  const [messageInput, setMessageInput] = useState('');
  const [clickedChips, setClickedChips] = useState([]);
  const chatContainerRef = useRef(null);

  const handleChange = (event) => {
    setMessageInput(event.target.value);
  };

  const handleKeyPress = (event) => {
    if (event.key === 'Enter') {
      sendMessage();
    }
  };

  const sendMessage = () => {
    if (messageInput.trim() !== '') {
      onSendMessage(messageInput);
      setMessageInput('');
    }
  };

  // Function to handle click on intent chips
  const handleIntentClick = (intent) => {
    onSendMessage(intent); // Send the intent as a message
    setClickedChips([...clickedChips, intent]); // Add the clicked chip to the state
  };

  useEffect(() => {
    // Scroll to the bottom of the chat container when messages change
    chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
  }, [messages]);

  return (
    <Paper
      style={{
        elevation: 6,
        position: 'absolute',
        left: '50%',
        top: '50%',
        transform: 'translate(-50%, -50%)',
        padding: 20,
        borderRadius: 20,
        width: '30%',
        height: '70%',
        overflow: 'hidden', // Hide scrollbars
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
      }}
    >
      <Typography variant="h5" gutterBottom>
        REIA - Renewable Energy Integration Assistant
      </Typography>
      <div
        ref={chatContainerRef}
        style={{
          marginBottom: 20,
          overflowY: 'auto',
          flexGrow: 1,
          marginRight: -20, // Compensate for scrollbar width
          paddingRight: 20, // Add padding to keep content from being obscured by scrollbar
        }}
      >
        {messages.map((message, index) => (
          <ChatMessage key={index} message={message} />
        ))}
      </div>
      <div style={{ marginBottom: 10 }}>
        {/* Render intent chips that haven't been clicked */}
        {['Greeting', 'Query'].map((intent) => (
          !clickedChips.includes(intent) && (
            <Chip
              key={intent}
              label={intent}
              onClick={() => handleIntentClick(intent)}
              style={{ marginRight: 5, cursor: 'pointer' }}
            />
          )
        ))}
        {/* Add more intent chips as needed */}
      </div>
      <div style={{ display: 'flex', alignItems: 'center' }}>
        <TextField
          label="Type a message"
          variant="outlined"
          value={messageInput}
          onChange={handleChange}
          onKeyPress={handleKeyPress}
          style={{ marginRight: 10, flex: 1 }}
        />
        <Button variant="contained" color="primary" onClick={sendMessage}>
          Send
        </Button>
      </div>
    </Paper>
  );
};

export default ChatWindow;
