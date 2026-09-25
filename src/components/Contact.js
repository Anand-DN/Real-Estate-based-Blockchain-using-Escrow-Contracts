import { useState } from 'react';

import close from '../assets/close.svg';

const Contact = ({ seller, toggleContact, setNotification }) => {
    const [name, setName] = useState('')
    const [email, setEmail] = useState('')
    const [message, setMessage] = useState('')

    const submitHandler = (e) => {
        e.preventDefault()
        setNotification('Message sent', 'success')
        toggleContact()
    }

    return (
        <div className="contact">
            <div className="contact__details">
                <div className="contact__card">
                    <div className="contact__avatar">A</div>
                    <h2>Alex Morgan</h2>
                    <p>Realty Agent</p>
                    <p>
                        <strong>Phone:</strong> +1 (555) 123-4567
                    </p>
                    <p>
                        <strong>Email:</strong> alex.morgan@millow.io
                    </p>
                    <p className="contact__address">
                        <strong>Seller:</strong> {seller && seller.slice(0, 6) + '...' + seller.slice(38, 42)}
                    </p>
                </div>

                <form className="contact__form" onSubmit={submitHandler}>
                    <h2>Send a message</h2>
                    <input
                        type="text"
                        placeholder="Your name"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        required
                    />
                    <input
                        type="email"
                        placeholder="Your email"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        required
                    />
                    <textarea
                        placeholder="Your message"
                        value={message}
                        onChange={(e) => setMessage(e.target.value)}
                        required
                    ></textarea>
                    <button type="submit" className="contact__submit">Send message</button>
                </form>

                <button onClick={toggleContact} className="home__close">
                    <img src={close} alt="Close" />
                </button>
            </div>
        </div>
    );
}

export default Contact;
