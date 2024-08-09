
let form = window.getElementById('user-query-form');
let input = window.getElementById('user-query-input');
console.log('OK')
input.addEventListener('keydown', (event) => {
  if (event.keyCode === '13') {
    event.preventDefault(); // prevent default behavior
    form.submit(); // submit the form
  }
});