import React from 'react';
import { Typography, Avatar } from '@mui/material';

const ChatMessage = ({ message }) => {
  return (
    <div
      key={message.text}
      style={{
        display: 'flex',
        alignItems: 'center',
        marginBottom: 10,
      }}
    >
      {!message.isUserMessage && message.withAvatar && (
        <Avatar alt="Bot Avatar" src="my-chatbot-app/public/avatar.png" style={{ marginRight: 10 }} />
      )}
      <div
        style={{
          backgroundColor: message.isUserMessage ? '#E3F2FD' : '#BBDEFB',
          padding: 10,
          borderRadius: 10,
          maxWidth: '70%',
        }}
      >
        <Typography variant="body1">{message.text}</Typography>
      </div>
      {message.isUserMessage && (
        <Avatar alt="User Avatar" src='my-chatbot-app/public/avatar.png' style={{ marginLeft: 10 }} />
      )}
    </div>
  );
};

export default ChatMessage;
