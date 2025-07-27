// Animation des étoiles
function createStars() {
    const starsContainer = document.querySelector('.stars');
    if (!starsContainer) return;
    
    const numStars = 100;

    for (let i = 0; i < numStars; i++) {
        const star = document.createElement('div');
        star.className = 'star';
        star.style.left = Math.random() * 100 + '%';
        star.style.top = Math.random() * 100 + '%';
        star.style.animationDelay = Math.random() * 2 + 's';
        starsContainer.appendChild(star);
    }
}

// Système de filtrage des projets
function initializeFilters() {
    const filterTabs = document.querySelectorAll('.filter-tab');
    const projectCards = document.querySelectorAll('.project-card');

    if (filterTabs.length === 0) return;

    filterTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Retirer la classe active de tous les onglets
            filterTabs.forEach(t => t.classList.remove('active'));
            // Ajouter la classe active à l'onglet cliqué
            tab.classList.add('active');

            const filter = tab.dataset.filter;

            projectCards.forEach(card => {
                if (filter === 'all' || card.dataset.category === filter) {
                    card.style.display = 'block';
                    setTimeout(() => {
                        card.style.opacity = '1';
                        card.style.transform = 'translateY(0)';
                    }, 100);
                } else {
                    card.style.opacity = '0';
                    card.style.transform = 'translateY(20px)';
                    setTimeout(() => {
                        card.style.display = 'none';
                    }, 300);
                }
            });
        });
    });
}

// Animation au scroll
function initializeScrollAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    });

    // Observer les cartes de projet
    document.querySelectorAll('.project-card').forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = 'all 0.6s ease';
        observer.observe(card);
    });

    // Observer les sections
    document.querySelectorAll('.section').forEach(section => {
        section.style.opacity = '0';
        section.style.transform = 'translateY(30px)';
        section.style.transition = 'all 0.8s ease';
        observer.observe(section);
    });
}

// Navigation active
function initializeNavigation() {
  const currentPage = window.location.pathname.split('/').pop() || 'index.html';
  const navLinks = document.querySelectorAll('.nav-link');

  navLinks.forEach(link => {
    const href = link.getAttribute('href').split('/').pop();

    if (
      href === currentPage ||
      (currentPage === '' && href === 'index.html') ||
      (currentPage === 'index.html' && href === 'index.html')
    ) {
      link.classList.add('active');
    } else {
      link.classList.remove('active');
    }
  });
}

// Appel après que le DOM est prêt
document.addEventListener('DOMContentLoaded', initializeNavigation);


// Effet de parallaxe léger sur les étoiles
function initializeParallax() {
    const stars = document.querySelector('.stars');
    if (!stars) return;

    window.addEventListener('mousemove', (e) => {
        const mouseX = e.clientX / window.innerWidth;
        const mouseY = e.clientY / window.innerHeight;
        
        stars.style.transform = `translate(${mouseX * 10}px, ${mouseY * 10}px)`;
    });
}

// Animation d'apparition des sections
function initializeSectionAnimations() {
    const sections = document.querySelectorAll('.section');
    
    sections.forEach((section, index) => {
        section.style.animationDelay = `${index * 0.2}s`;
    });
}

// Smooth scroll pour les liens d'ancrage
function initializeSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Initialisation globale
document.addEventListener('DOMContentLoaded', () => {
    createStars();
    initializeFilters();
    initializeScrollAnimations();
    initializeContactForm();
    initializeNavigation();
    initializeParallax();
    initializeSectionAnimations();
    initializeSmoothScroll();
});

// Gestion du redimensionnement de la fenêtre
window.addEventListener('resize', () => {
    // Recalculer les positions si nécessaire
});

// Fonction utilitaire pour déboguer
function debugInfo() {
    console.log('🌟 Portfolio Website Loaded');
    console.log('📄 Current page:', window.location.pathname);
    console.log('🎯 Stars created:', document.querySelectorAll('.star').length);
    console.log('🎨 Sections found:', document.querySelectorAll('.section').length);
}