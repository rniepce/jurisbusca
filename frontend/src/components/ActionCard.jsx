import React from 'react';
import { FaArrowRight } from 'react-icons/fa';
import './ActionCard.css'; // We'll create this or use global CSS

const ActionCard = ({ icon, text, onClick }) => {
    return (
        <button className="action-card" onClick={onClick}>
            <div className="icon-box">
                {icon}
            </div>
            <span className="action-text">{text}</span>
            <FaArrowRight className="action-arrow" />
        </button>
    );
};

export default ActionCard;
