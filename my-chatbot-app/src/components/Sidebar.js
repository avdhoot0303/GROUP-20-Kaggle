import React from 'react';
import { Typography, Paper, Box } from '@mui/material';

const Sidebar = () => {
  // Define your sidebar content here
  const sidebarContent = [
    { title: 'Card 1', content: 'This is the content of Card 1' },
    { title: 'Card 2', content: 'This is the content of Card 2' },
    { title: 'Card 3', content: 'This is the content of Card 3' },
  ];

  return (
    <Box
      style={{
        position: 'absolute',
        right: 60,
        top: '50%',
        transform: 'translateY(-50%)',
        padding: 10,
        width: '15%',
        height: '55%',
        display: 'flex',
        flexDirection: 'column',
        border: '1px solid #ccc',
        borderRadius: 20
          }}
    >
      <Typography variant="h6" gutterBottom>
        Sidebar
      </Typography>
      {sidebarContent.map((card, index) => (
        <Paper key={index} style={{ padding: 10, marginBottom: 10 }}>
          <Typography variant="h6">{card.title}</Typography>
          <Typography>{card.content}</Typography>
        </Paper>
      ))}
    </Box>
  );
};

export default Sidebar;
