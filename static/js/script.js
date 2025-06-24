document.addEventListener('DOMContentLoaded', function() {
    // Form functionality
    const form = document.getElementById('sizeForm');
    const submitBtn = document.getElementById('submitBtn');
    const loading = document.getElementById('loading');
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    const inputs = form.querySelectorAll('input, select');

    // Innovation features
    const brandPreview = document.getElementById('brandPreview');
    const itemPreview = document.getElementById('itemPreview');

    // Form validation and smart progress bar
    function updateProgress() {
        const filledInputs = Array.from(inputs).filter(input => input.value.trim() !== '').length;
        const progress = (filledInputs / inputs.length) * 100;
        progressFill.style.width = progress + '%';
        progressText.textContent = `${Math.round(progress)}% Complete`;
        
        // Update step indicators
        const stepIndicators = document.querySelectorAll('.step-indicator');
        stepIndicators.forEach((indicator, index) => {
            if (progress >= (index + 1) * 33.33) {
                indicator.classList.add('active');
            } else {
                indicator.classList.remove('active');
            }
        });
        
        if (progress === 100) {
            submitBtn.disabled = false;
            submitBtn.style.opacity = '1';
        } else {
            submitBtn.disabled = true;
            submitBtn.style.opacity = '0.6';
        }
    }

    // Add event listeners for real-time validation
    inputs.forEach(input => {
        input.addEventListener('input', updateProgress);
        input.addEventListener('change', updateProgress);
    });

    // Brand and Item preview functionality
    const brandSelect = document.getElementById('brand');
    const itemSelect = document.getElementById('item');

    brandSelect.addEventListener('change', function() {
        const selectedBrand = this.value;
        if (selectedBrand) {
            brandPreview.textContent = `Selected: ${selectedBrand}`;
            brandPreview.classList.add('show');
        } else {
            brandPreview.classList.remove('show');
        }
    });

    itemSelect.addEventListener('change', function() {
        const selectedItem = this.value;
        if (selectedItem) {
            itemPreview.textContent = `Selected: ${selectedItem}`;
            itemPreview.classList.add('show');
        } else {
            itemPreview.classList.remove('show');
        }
    });

    // Form submission with enhanced loading animation
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Show loading animation
        submitBtn.style.display = 'none';
        loading.style.display = 'block';
        
        // Animate loading steps
        const steps = document.querySelectorAll('.step');
        let currentStep = 0;
        
        const stepInterval = setInterval(() => {
            if (currentStep < steps.length) {
                steps.forEach((step, index) => {
                    if (index === currentStep) {
                        step.classList.add('active');
                    } else {
                        step.classList.remove('active');
                    }
                });
                currentStep++;
            } else {
                clearInterval(stepInterval);
                form.submit();
            }
        }, 800);
    });

    // Enhanced interactive effects
    inputs.forEach(input => {
        input.addEventListener('focus', function() {
            this.parentElement.style.transform = 'scale(1.02)';
            this.parentElement.style.boxShadow = '0 5px 15px rgba(102, 126, 234, 0.2)';
        });
        
        input.addEventListener('blur', function() {
            this.parentElement.style.transform = 'scale(1)';
            this.parentElement.style.boxShadow = 'none';
        });
    });

    // Notification system
    function showNotification(message) {
        // Create notification container if it doesn't exist
        let notificationContainer = document.getElementById('notification-container');
        if (!notificationContainer) {
            notificationContainer = document.createElement('div');
            notificationContainer.id = 'notification-container';
            notificationContainer.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 10000;
                display: flex;
                flex-direction: column;
                gap: 10px;
            `;
            document.body.appendChild(notificationContainer);
        }

        // Create notification element
        const notification = document.createElement('div');
        notification.style.cssText = `
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 15px 20px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            transform: translateX(100%);
            transition: all 0.3s ease;
            max-width: 300px;
            word-wrap: break-word;
        `;
        
        notification.innerHTML = `
            <div style="display: flex; align-items: center; gap: 10px;">
                <i class="fas fa-info-circle"></i>
                <span>${message}</span>
            </div>
        `;
        
        notificationContainer.appendChild(notification);
        
        // Animate in
        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 100);
        
        // Auto remove after 3 seconds
        setTimeout(() => {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }

    // Touch gestures for mobile
    let touchStartY = 0;
    let touchEndY = 0;

    document.addEventListener('touchstart', function(e) {
        touchStartY = e.changedTouches[0].screenY;
    });

    document.addEventListener('touchend', function(e) {
        touchEndY = e.changedTouches[0].screenY;
        handleSwipe();
    });

    function handleSwipe() {
        const swipeThreshold = 50;
        const diff = touchStartY - touchEndY;

        if (Math.abs(diff) > swipeThreshold) {
            if (diff > 0) {
                // Swipe up - focus on first input
                const firstInput = document.querySelector('input');
                if (firstInput) {
                    firstInput.focus();
                    showNotification('Focus on first field');
                }
            } else {
                // Swipe down - scroll to submit button
                const submitBtn = document.getElementById('submitBtn');
                submitBtn.scrollIntoView({ behavior: 'smooth' });
                showNotification('Scroll to submit button');
            }
        }
    }

    // Initialize progress on page load
    updateProgress();
}); 