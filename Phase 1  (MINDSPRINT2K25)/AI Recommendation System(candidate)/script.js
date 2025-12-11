/* =========================
   Smooth Scroll
========================= */
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener("click", function (e) {
    e.preventDefault();
    document.querySelector(this.getAttribute("href")).scrollIntoView({
      behavior: "smooth"
    });
  });
});

/* =========================
   Typing Effect (Hero Text)
========================= */
const heroTitle = document.querySelector(".hero h1");
const text = "Find Your Dream Internship 🚀";
let index = 0;

function typeEffect() {
  if (index < text.length) {
    heroTitle.textContent += text.charAt(index);
    index++;
    setTimeout(typeEffect, 100);
  }
}
heroTitle.textContent = ""; // clear initial
typeEffect();

/* =========================
   Auto Sliding Featured Companies
========================= */
const companyContainer = document.querySelector(".company-cards");

if (companyContainer) {
  let scrollAmount = 0;
  setInterval(() => {
    scrollAmount += 250; // px scroll step
    if (scrollAmount >= companyContainer.scrollWidth) {
      scrollAmount = 0;
    }
    companyContainer.scrollTo({
      left: scrollAmount,
      behavior: "smooth"
    });
  }, 3000); // every 3 sec
}

/* =========================
   Animate on Scroll (Fade-in)
========================= */
const faders = document.querySelectorAll(
  ".stat-card, .company-card, .trending-card, .category-card, .step, .testimonial-card"
);

const appearOptions = {
  threshold: 0.2,
  rootMargin: "0px 0px -50px 0px"
};

const appearOnScroll = new IntersectionObserver(function (entries, appearOnScroll) {
  entries.forEach(entry => {
    if (!entry.isIntersecting) return;
    entry.target.classList.add("fade-in");
    appearOnScroll.unobserve(entry.target);
  });
}, appearOptions);

faders.forEach(fader => {
  appearOnScroll.observe(fader);
});

/* =========================
   Particle Background
========================= */
particlesJS("particles-js", {
  particles: {
    number: { value: 80, density: { enable: true, value_area: 800 } },
    color: { value: "#ffffff" },
    shape: { type: "circle" },
    opacity: { value: 0.5, random: false },
    size: { value: 3, random: true },
    line_linked: {
      enable: true,
      distance: 150,
      color: "#ffffff",
      opacity: 0.4,
      width: 1
    },
    move: { enable: true, speed: 2, direction: "none", out_mode: "out" }
  },
  interactivity: {
    detect_on: "canvas",
    events: { onhover: { enable: true, mode: "repulse" } },
    modes: { repulse: { distance: 100, duration: 0.4 } }
  },
  retina_detect: true
});

