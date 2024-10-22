// Toggle Password Visibility
const togglePassword = document.querySelector('#togglePassword');
const password = document.querySelector('#password');
togglePassword.addEventListener('click', function () {
    const type = password.getAttribute('type') === 'password' ? 'text' : 'password';
    password.setAttribute('type', type);
    this.textContent = this.textContent === 'Show' ? 'Hide' : 'Show';
});

// Geolocation (if enabled)
const geolocationCheckbox = document.querySelector('#enable_geolocation');
geolocationCheckbox.addEventListener('change', function () {
    if (this.checked) {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(function (position) {
                console.log("Latitude: " + position.coords.latitude + ", Longitude: " + position.coords.longitude);
                // Do not store the location, just for user info
            });
        } else {
            console.log("Geolocation is not supported by this browser.");
        }
    }
});
