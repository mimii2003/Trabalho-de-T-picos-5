function login() {
    document.getElementById('loginBtn').addEventListener('click', function () {
        document.getElementById('authButtons').classList.add('d-none');
        document.getElementById('profileIcon').classList.remove('d-none');
    });
}