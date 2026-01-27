export const PROGRESS_KEY = 'tutorial_platform_progress';

export const getCompletedTutorials = () => {
    try {
        const stored = localStorage.getItem(PROGRESS_KEY);
        return stored ? JSON.parse(stored) : [];
    } catch (e) {
        console.error("Error reading progress:", e);
        return [];
    }
};

export const isTutorialCompleted = (id) => {
    const completed = getCompletedTutorials();
    return completed.includes(id);
};

export const toggleTutorialCompletion = (id) => {
    const completed = getCompletedTutorials();
    let newCompleted;

    if (completed.includes(id)) {
        newCompleted = completed.filter(item => item !== id);
    } else {
        newCompleted = [...completed, id];
    }

    localStorage.setItem(PROGRESS_KEY, JSON.stringify(newCompleted));

    // Dispatch a custom event so other components can react immediately
    window.dispatchEvent(new Event('progressUpdated'));

    return !completed.includes(id); // Returns true if now completed, false otherwise
};
