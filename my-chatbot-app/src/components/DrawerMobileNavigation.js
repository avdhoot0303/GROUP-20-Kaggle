import React from 'react';
import { IconButton, Drawer, Box, Typography, Input, List, ListItemButton } from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import SearchIcon from '@mui/icons-material/Search';
import CloseIcon from '@mui/icons-material/Close';
import '../App.css'; // Import CSS file for styling

const DrawerMobileNavigation = () => {
  const [open, setOpen] = React.useState(false);

  return (
    <React.Fragment>
      <Drawer
        anchor="left"
        open={open}
        onClose={() => setOpen(false)}
        sx={{ display: { sm: 'none', md: 'block' } }}
      >
        <Box
          sx={{
            width: 250,
            bgcolor: 'primary.main',
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
          }}
        >
          <Box sx={{ py: 1 }}>
            <Typography variant="h6" color="inherit">
              Menu
            </Typography>
          </Box>
          <Input
            size="small"
            placeholder="Search"
            variant="standard"
            endAdornment={<SearchIcon />}
            sx={{ m: 2 }}
          />
          <List sx={{ textAlign: 'left' }}>
            <ListItemButton>
              <Typography variant="body1">History</Typography>
            </ListItemButton>
            <ListItemButton>
              <Typography variant="body1">About</Typography>
            </ListItemButton>
          </List>
        </Box>
      </Drawer>
      <IconButton variant="outlined" color="inherit" onClick={() => setOpen(true)} sx={{ ml: 1 }}>
        <MenuIcon />
      </IconButton>
    </React.Fragment>
  );
};

export default DrawerMobileNavigation;
