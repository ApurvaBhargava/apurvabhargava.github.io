document.addEventListener("DOMContentLoaded", function () {
  // Theme toggle
  const body = document.body;
  const toggle = document.getElementById("theme-toggle");
  const savedTheme = window.localStorage.getItem("theme");

  if (savedTheme === "dark") {
    body.classList.remove("theme-light");
    body.classList.add("theme-dark");
  } else if (savedTheme === "light") {
    body.classList.add("theme-light");
    body.classList.remove("theme-dark");
  } else {
    body.classList.add("theme-light");
  }

  if (toggle) {
    toggle.addEventListener("click", () => {
      const isDark = body.classList.toggle("theme-dark");
      body.classList.toggle("theme-light", !isDark);
      window.localStorage.setItem("theme", isDark ? "dark" : "light");
    });
  }

  // Mobile nav toggle
  const navToggle = document.getElementById("nav-toggle");
  const navLinks = document.getElementById("nav-links");

  if (navToggle && navLinks) {
    navToggle.addEventListener("click", () => {
      navLinks.classList.toggle("open");
    });
  }
});
