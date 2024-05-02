function updateWindows() {
    // Update lane masked image
    document.getElementById('laneMaskedImage').innerHTML = '<img src="/lane_masked_image">';
    
    // Update obstacle masked image
    document.getElementById('obstacleMaskedImage').innerHTML = '<img src="/obstacle_masked_image">';
    
    // Update depth colormap
    document.getElementById('depthColormap').innerHTML = '<img src="/depth_colormap">';
}

// Update every 5 seconds
setInterval(updateWindows, 5000);
