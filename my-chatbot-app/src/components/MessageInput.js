import React, { useState } from 'react';
import { Paper, Typography, Avatar } from '@mui/material';
import MessageInput from './MessageInput'; // Assuming MessageInput is a separate component

const ChatMessage = ({ message }) => {
  return (
    <div
      key={message.text}
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: message.isUserMessage ? 'flex-end' : 'flex-start',
        marginBottom: 10,
      }}
    >
      <div
        sx={{
          backgroundColor: message.isUserMessage ? '#E3F2FD' : '#BBDEFB',
          padding: 10,
          borderRadius: 10,
          maxWidth: '70%',
          animation: message.isUserMessage ? 'fadeInRight 0.5s' : 'fadeInLeft 0.5s',
        }}
      >
        <Typography variant="body1">{message.text}</Typography>
      </div>
      {message.isUserMessage && (
        <Avatar alt="User Avatar" src="https://via.placeholder.com/40" style={{ marginLeft: 10 }} />
      )}
    </div>
  );
};

const ChatWindow = ({ messages }) => {
  const [isLoading, setIsLoading] = useState(false);

  const handleSendMessage = (message) => {
    // You can handle sending the new message to your chat system here
    // For demonstration, let's simulate a delay before receiving a response
    setIsLoading(true);
    setTimeout(() => {
      const responseMessage = {
        text: 'This is a response to your message: ' + message,
        isUserMessage: false, // Assuming bot/system messages here
      };
      // Assuming you have a function to update the chat messages
      // updateMessages([...messages, responseMessage]);
      setIsLoading(false);
    }, 1500); // Simulating a 1.5 second delay
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginTop: 50 }}>
      <Typography variant="h4" gutterBottom={true} style={{ marginBottom: 20 }}>
        REIA - Renewable Energy Integration Assistant
      </Typography>
      <Paper elevation={3} style={{ padding: 20, width: '60%', minHeight: 300, overflowY: 'auto', borderRadius: 10 }}>
        {messages.map((message, index) => (
          <ChatMessage key={index} message={message} />
        ))}
        <MessageInput onSendMessage={handleSendMessage} isLoading={isLoading} />
      </Paper>
    </div>
  );
};

export default ChatWindow;
