// Toggle Password Visibility
const togglePassword = document.querySelector('#togglePassword');
const password = document.querySelector('#password');
togglePassword.addEventListener('click', function () {
    const type = password.getAttribute('type') === 'password' ? 'text' : 'password';
    password.setAttribute('type', type);
    this.textContent = this.textContent === 'Show' ? 'Hide' : 'Show';
});

const toggleConfirmPassword = document.querySelector('#toggleConfirmPassword');
const confirmPassword = document.querySelector('#confirm_password');
toggleConfirmPassword.addEventListener('click', function () {
    const type = confirmPassword.getAttribute('type') === 'password' ? 'text' : 'password';
    confirmPassword.setAttribute('type', type);
    this.textContent = this.textContent === 'Show' ? 'Hide' : 'Show';
});
