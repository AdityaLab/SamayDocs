// Make all external links open in a new tab
document.addEventListener('DOMContentLoaded', function() {
  var links = document.querySelectorAll('a[href^="http"]');
  
  links.forEach(function(link) {
    var currentHost = window.location.hostname;
    var linkHost = link.hostname;
    
    // Check if link is external (not same domain)
    if (linkHost !== currentHost && linkHost !== '') {
      link.setAttribute('target', '_blank');
      link.setAttribute('rel', 'noopener noreferrer');
    }
  });
});

