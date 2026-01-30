
import json
import os

# PATHS
EXAM_FILE = os.path.join("backend", "data", "exam_questions.json")

# FULL EXPERT CONTENT FOR PYTHON UNIT 4
PYTHON_UNIT_4_DATA = {
    "unit": "Python Unit 4: Network & GUI",
    "sections": {
        "Part A (1-Mark)": [
            {"question": "What is a Socket?", "answer": "One endpoint of a two-way communication link between two programs running on the network."},
            {"question": "Define TCP.", "answer": "Transmission Control Protocol: A connection-oriented protocol that ensures reliable order of packet delivery."},
            {"question": "Define UDP.", "answer": "User Datagram Protocol: A connectionless protocol that sends packets without checking for delivery (faster)."},
            {"question": "What is an IP Address?", "answer": "A numerical label assigned to each device connected to a computer network (e.g., 192.168.1.1)."},
            {"question": "What is a Port Number?", "answer": "A logical construct that identifies a specific process or service on a network device (e.g., Port 80 for HTTP)."},
            {"question": "What is 'socket.bind()'? method?", "answer": "A method used to associate a socket with a specific network interface and port number."},
            {"question": "What is 'socket.listen()'? method?", "answer": "A method that enables a server to accept connections from clients."},
            {"question": "What does 'accept()' return?", "answer": "It returns a new socket object representing the connection and the address of the client."},
            {"question": "What is Tkinter?", "answer": "The standard Python interface to the Tk GUI toolkit, used for building desktop applications."},
            {"question": "Define GUI.", "answer": "Graphical User Interface: Allows users to interact with electronic devices through graphical icons."},
            {"question": "What is a Widget?", "answer": "A basic building block of a GUI, such as a Button, Label, or Entry field."},
            {"question": "What is a Layout Manager?", "answer": "A mechanism in Tkinter to arrange widgets on the screen (Pack, Grid, Place)."},
            {"question": "What is 'mainloop()'?", "answer": "An infinite loop in Tkinter that runs the application, waits for events, and processes them."},
            {"question": "Define Event Handling.", "answer": "The mechanism that controls the event (like a click) and decides what should happen if an event occurs."},
            {"question": "What is a Canvas?", "answer": "A widget used to draw shapes (lines, ovals, polygons) or display images in Tkinter."},
            {"question": "What is the 'Entry' widget?", "answer": "A widget used to accept single-line text input from the user."},
            {"question": "What is the 'Label' widget?", "answer": "A widget used to display text or images that the user cannot edit."},
            {"question": "What is 'smtplib'?", "answer": "A Python module that defines an SMTP client session object for sending emails."},
            {"question": "Define SMTP.", "answer": "Simple Mail Transfer Protocol: The standard protocol for email transmission."},
            {"question": "What represents localhost?", "answer": "The IP address 127.0.0.1 or the hostname 'localhost'."}
        ],
        "Part B (2-Marks)": [
            {"question": "TCP vs UDP.", "answer": "• TCP: Reliable, Ordered, Connection-oriented (Email, Web).\n• UDP: Unreliable, Unordered, Connectionless (Gaming, Streaming)."},
            {"question": "Server vs Client.", "answer": "• Server: Provides a service, listens for requests.\n• Client: Consumes a service, initiates requests."},
            {"question": "Pack vs Grid Geometry Manager.", "answer": "• Pack: Places widgets in blocks relative to each other (Top, Bottom).\n• Grid: Places widgets in a 2D table (Row, Column)."},
            {"question": "What is 'socket.recv()'? method", "answer": "• Receives data from the socket.\n• Must specify buffer size (e.g., 1024 bytes). Returns bytes object."},
            {"question": "Explain 'geometry()' method.", "answer": "• Sets the size and position of the main window.\n• Syntax: window.geometry('300x200')."},
            {"question": "Button vs Checkbutton.", "answer": "• Button: Triggers an action when clicked.\n• Checkbutton: Toggles a boolean state (On/Off)."},
            {"question": "What is 'AF_INET'?", "answer": "• Address Family for IPv4.\n• Used when creating a socket (socket.AF_INET)."},
            {"question": "What is 'SOCK_STREAM'?", "answer": "• Socket Type for TCP.\n• Ensures reliable, sequential data flow."},
            {"question": "List 4 Tkinter Widgets.", "answer": "• Label\n• Button\n• Entry\n• Text"},
            {"question": "Purpose of 'filedialog'.", "answer": "• A module to open system file dialogs (Open, Save As).\n• Allows user to select files from disk."},
            {"question": "What is a Treeview?", "answer": "• A widget to display hierarchical data (like file explorer).\n• Supports columns and rows."},
            {"question": "What is 'messagebox'?", "answer": "• A module to display pop-up messages.\n• Examples: showinfo, showwarning, askyesno."},
            {"question": "Binding events in Tkinter.", "answer": "• Connecting an event (e.g., <Button-1>) to a function.\n• Syntax: widget.bind('<Event>', handler_function)."},
            {"question": "What is Port 80 and 443?", "answer": "• Port 80: Default for HTTP (Unsecured Web).\n• Port 443: Default for HTTPS (Secured Web)."},
            {"question": "Close() method usage.", "answer": "• Releases the resource associated with the socket or file.\n• Essential to prevent resource leaks."}
        ],
        "Part C (3-Marks)": [
            {"question": "Steps to create a TCP Server.", "answer": "1. **Definition:** Server Setup.\n2. **Explanation:** 1. Create Socket. 2. Bind to IP/Port. 3. Listen. 4. Accept Connection. 5. Send/Recv Data. 6. Close.\n3. **Example:** server.bind(('localhost', 9999))."},
            {"question": "Steps to create a TCP Client.", "answer": "1. **Definition:** Client Setup.\n2. **Explanation:** 1. Create Socket. 2. Connect to Server IP/Port. 3. Send/Recv Data. 4. Close.\n3. **Example:** client.connect(('localhost', 9999))."},
            {"question": "Tkinter Application Structure.", "answer": "1. **Definition:** Boilerplate code.\n2. **Explanation:** 1. Import tkinter. 2. Create Root window. 3. Add Widgets. 4. Start Mainloop.\n3. **Example:** root = Tk(); root.mainloop()."},
            {"question": "Grid Layout Logic.", "answer": "1. **Definition:** Table-based layout.\n2. **Explanation:** Widgets placed at (row, col). Can span multiple rows/cols (rowspan, columnspan). Sticky param aligns content.\n3. **Example:** btn.grid(row=0, column=1)."},
            {"question": "Sending Email via Python.", "answer": "1. **Definition:** Automation.\n2. **Explanation:** Use `smtplib`. 1. Connect to SMTP server (Gmail). 2. Login. 3. Sendmail(From, To, Msg). 4. Quit.\n3. **Example:** s.sendmail('me', 'you', 'Subject: Hi')."},
            {"question": "Handling Button Clicks.", "answer": "1. **Definition:** Interactivity.\n2. **Explanation:** Define a function. Pass function name to `command=` parameter of Button. DO NOT use parentheses in command.\n3. **Example:** Button(text='Click', command=my_func)."},
            {"question": "Entry Widget usage.", "answer": "1. **Definition:** Input field.\n2. **Explanation:** User Types text. Program gets text using `.get()`. Program sets text using `.insert()`. Deletes using `.delete()`.\n3. **Example:** name = entry_box.get()."},
            {"question": "Socket Error Handling.", "answer": "1. **Definition:** Robustness.\n2. **Explanation:** Network can fail. Use try-except blocks. Catch `socket.error` or `ConnectionRefusedError`.\n3. **Example:** try: s.connect() except: print('Server Down')."},
            {"question": "File Transfer Logic.", "answer": "1. **Definition:** Sending files over socket.\n2. **Explanation:** Sender: Open file, read bytes, sendall() chunks. Receiver: Recv loops, write bytes to file.\n3. **Example:** sending 1024 byte chunks until file ends."},
            {"question": "Menu Bar Creation.", "answer": "1. **Definition:** GUI Navigation.\n2. **Explanation:** Create Menu object. Add Cascades (File, Edit). Add Commands (Open, Exit). Config root menu.\n3. **Example:** file_menu.add_command(label='Exit', command=root.quit)."},
            {"question": "Frame Widget usage.", "answer": "1. **Definition:** Container.\n2. **Explanation:** Groups other widgets together. Useful for complex layouts (e.g., Left Sidebar, Right Content).\n3. **Example:** frame = Frame(root); btn.pack(in_=frame)."},
            {"question": "Blocking vs Non-Blocking Sockets.", "answer": "1. **Definition:** execution mode.\n2. **Explanation:** Blocking: Program waits (hangs) at recv() until data arrives. Non-Blocking: Returns error immediately if no data.\n3. **Example:** Blocking is simpler; Non-Blocking needs loops/select()."},
            {"question": "Checkbutton vs Radiobutton.", "answer": "1. **Definition:** Selection widgets.\n2. **Explanation:** Checkbutton: Multiple selections allowed (Square). Radiobutton: Only one selection allowed (Circle).\n3. **Example:** Pizza Toppings (Check) vs Gender (Radio)."},
            {"question": "Canvas Drawing.", "answer": "1. **Definition:** Graphics area.\n2. **Explanation:** Use coordinates (x1, y1, x2, y2). Methods: create_line, create_oval, create_rectangle.\n3. **Example:** c.create_oval(10, 10, 50, 50, fill='red')."},
            {"question": "Dialog Boxes.", "answer": "1. **Definition:** User prompts.\n2. **Explanation:** `messagebox.showinfo` (Alert), `messagebox.askyesno` (Confirmation), `filedialog.askopenfilename`.\n3. **Example:** if askyesno('Quit?'): root.destroy()."}
        ],
        "Part D (5-Marks)": [
            {
                "question": "Detailed Explanation of Socket Programming.",
                "answer": "1. **Definition:**\n   A way of connecting two nodes on a network to communicate with each other.\n\n2. **Goal:**\n   To enable data exchange between a Server (Listener) and a Client (Requester).\n\n3. **Core Concept:**\n   Socket = IP Address + Port Number. Uses TCP (Reliable) or UDP (Fast).\n\n4. **Technique / Method:**\n   Server: `socket()` -> `bind()` -> `listen()` -> `accept()`. Client: `socket()` -> `connect()`.\n\n5. **Applications:**\n   Web Browsers (HTTP), Chat Applications, Online Gaming, File FTP."
            },
            {
                "question": "Detailed Explanation of TCP Server Implementation.",
                "answer": "1. **Definition:**\n   Writing a Python script that creates a specific service listening on a port.\n\n2. **Goal:**\n   To accept incoming connections from clients and process their requests.\n\n3. **Core Concept:**\n   The server must run continuously (while loop). It creates a new dedicated socket for each client.\n\n4. **Technique / Method:**\n   1. `s = socket.socket()`. 2. `s.bind((HOST, PORT))`. 3. `s.listen(5)`. 4. `conn, addr = s.accept()`. 5. `conn.send()`. 6. `conn.close()`.\n\n5. **Applications:**\n   Web Servers (Apache/Nginx logic), Database Servers, Echo Server."
            },
            {
                "question": "Detailed Explanation of TCP Client Implementation.",
                "answer": "1. **Definition:**\n   Writing a Python script that connects to a known server to request data.\n\n2. **Goal:**\n   To initiate communication and consume the service provided by the server.\n\n3. **Core Concept:**\n   Active Open. The client 'dials' the server's IP and Port.\n\n4. **Technique / Method:**\n   1. `s = socket.socket()`. 2. `s.connect((HOST, PORT))`. 3. `s.send(b'Hello')`. 4. `data = s.recv(1024)`. 5. `s.close()`.\n\n5. **Applications:**\n   Web Scrapers, Chat Clients, Email Clients."
            },
            {
                "question": "Detailed Explanation of Tkinter Architecture.",
                "answer": "1. **Definition:**\n   Python's de-facto standard GUI (Graphical User Interface) package.\n\n2. **Goal:**\n   To provide a fast and easy way to create GUI applications.\n\n3. **Core Concept:**\n   Event-Driven Programming. The program waits in an event loop (mainloop) for user actions (clicks, keys).\n\n4. **Technique / Method:**\n   Hierarchy: Root Window -> Frames -> Widgets (Buttons, Entries). Geometry Managers control placement.\n\n5. **Applications:**\n   Calculator apps, Text Editors, Configuration Tools."
            },
            {
                "question": "Detailed Explanation of Geometry Managers.",
                "answer": "1. **Definition:**\n   Methods in Tkinter to organize and position widgets within the parent window.\n\n2. **Goal:**\n   To accept widget layout requests and manage the display coordinates.\n\n3. **Core Concept:**\n   Three types: Pack (Block), Grid (Table), Place (Absolute).\n\n4. **Technique / Method:**\n   Pack: `side='left', fill='x'`. Grid: `row=0, col=1`. Place: `x=50, y=100`. NEVER mix pack/grid in same frame.\n\n5. **Applications:**\n   Designing responsive or fixed layouts for desktop applications."
            },
            {
                "question": "Detailed Explanation of Email Sending (smtplib).",
                "answer": "1. **Definition:**\n   Using Python's built-in module 'smtplib' to send emails via the Simple Mail Transfer Protocol.\n\n2. **Goal:**\n   To automate email notifications or reports.\n\n3. **Core Concept:**\n   Client-Server Model. Python acts as a client connecting to an SMTP server (like smtp.gmail.com).\n\n4. **Technique / Method:**\n   1. `s = smtplib.SMTP('host', port)`. 2. `s.starttls()` (Security). 3. `s.login(user, pass)`. 4. `s.sendmail()`. 5. `s.quit()`.\n\n5. **Applications:**\n   Password reset emails, Daily automated reports, System alerts."
            },
            {
                "question": "Detailed Explanation of File Transfer Protocol Logic.",
                "answer": "1. **Definition:**\n   Moving files from one system to another over a network.\n\n2. **Goal:**\n   To replicate a file accurately on a remote machine.\n\n3. **Core Concept:**\n   Files are just streams of bytes. Read binary ('rb') -> Send -> Write binary ('wb').\n\n4. **Technique / Method:**\n   Sender: Loop { Read 1KB -> Send }. Receiver: Loop { Recv 1KB -> Write } until 0 bytes received.\n\n5. **Applications:**\n   Dropbox-like sync, Uploading images, Software updates."
            },
            {
                "question": "Detailed Explanation of UDP Communication.",
                "answer": "1. **Definition:**\n   Using connectionless User Datagram Protocol for communication.\n\n2. **Goal:**\n   To send data fast without the overhead of establishing a connection.\n\n3. **Core Concept:**\n   Fire and Forget. No guarantee of delivery or order. Packets are independent datagrams.\n\n4. **Technique / Method:**\n   `socket(AF_INET, SOCK_DGRAM)`. Use `sendto(data, addr)` and `recvfrom(bufsize)`.\n\n5. **Applications:**\n   Video Streaming (Netflix), VoIP (Skype calls), DNS lookups."
            },
            {
                "question": "Detailed Explanation of Event Handling in GUI.",
                "answer": "1. **Definition:**\n   The way a GUI application responds to user inputs like mouse clicks or key presses.\n\n2. **Goal:**\n   To make the application interactive and responsive.\n\n3. **Core Concept:**\n   Binding. Linking an Event (<Button-1>) to a Callback Function (handler).\n\n4. **Technique / Method:**\n   `widget.bind(event_sequence, callback)`. callback takes an `event` object argument containing mouse x,y etc.\n\n5. **Applications:**\n   Games (Arrow keys to move), Forms (Enter key to submit), Shortcuts (Ctrl+S)."
            },
            {
                "question": "Detailed Explanation of Treeview Widget.",
                "answer": "1. **Definition:**\n   A Tkinter widget used to display data in a hierarchical (tree) or tabular structure.\n\n2. **Goal:**\n   To visualize complex nested data or multi-column lists.\n\n3. **Core Concept:**\n   Nodes can have children. Columns allow displaying attributes of each node.\n\n4. **Technique / Method:**\n   `tree = ttk.Treeview()`. `tree.insert('', 'end', text='Parent')`. Define columns and headings.\n\n5. **Applications:**\n   File Explorers, Database Result views, Organization charts."
            }
        ]
    }
}

def populate_python_unit4():
    try:
        if not os.path.exists(EXAM_FILE):
             print("Error: exam_questions.json not found.")
             return

        with open(EXAM_FILE, 'r') as f:
            data = json.load(f)

        # 1. Remove ANY existing "Python Unit 4: Network & GUI"
        data = [u for u in data if "Python Unit 4: Network & GUI" not in u['unit']]
        
        # 2. Append the NEW Full Unit
        data.append(PYTHON_UNIT_4_DATA)
        print("Successfully replaced Python Unit 4 with 60 Expert Questions.")

        # 3. Save
        with open(EXAM_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        print("Saved to exam_questions.json.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    populate_python_unit4()
