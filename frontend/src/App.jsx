import React, { useState } from 'react';
import { Routes, Route, Outlet } from 'react-router-dom';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import HomePage from './pages/HomePage';
import TutorialPage from './pages/TutorialPage';
import EditorPage from './pages/EditorPage';
import CertificatePage from './pages/CertificatePage';
import ExamplesPage from './pages/ExamplesPage';
import ExercisesPage from './pages/ExercisesPage';
import QuizzesPage from './pages/QuizzesPage';
import ResourcesPage from './pages/ResourcesPage';
import QuestionsPage from './pages/QuestionsPage';



const MainLayout = () => {
    const [isSidebarOpen, setIsSidebarOpen] = useState(false);

    return (
        <div className="app-main">
            <Navbar toggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)} />
            <div className="app-container">
                <Sidebar isOpen={isSidebarOpen} closeSidebar={() => setIsSidebarOpen(false)} />
                <Outlet />
            </div>
        </div>
    );
};

const EditorLayout = () => {
    return (
        <div className="app-editor">
            {/* Editor might want a navbar too, or maybe not. Let's keep it clean. */}
            <Navbar />
            <Outlet />
        </div>
    );
}

function App() {
    return (
        <Routes>
            <Route element={<MainLayout />}>
                <Route path="/" element={<HomePage />} />
                <Route path="/tutorial/:id" element={<TutorialPage />} />
                <Route path="/certificate" element={<CertificatePage />} />
                <Route path="/examples" element={<ExamplesPage />} />
                <Route path="/exercises" element={<ExercisesPage />} />
                <Route path="/quizzes" element={<QuizzesPage />} />
                <Route path="/questions" element={<QuestionsPage />} />
                <Route path="/resources" element={<ResourcesPage />} />
            </Route>
            <Route element={<EditorLayout />}>
                <Route path="/editor" element={<EditorPage />} />
            </Route>
        </Routes>
    );
}

export default App;
